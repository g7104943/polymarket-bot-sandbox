#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT = Path('/Users/mac/polyfun')
REPORTS = ROOT / 'reports'
RAW = ROOT / 'data' / 'raw'
BASE_SCRIPT = ROOT / 'scripts' / 'ops' / 'run_crypto15m_1h_multasset_pressure_search_latest.py'
FILL_SCRIPT = ROOT / 'scripts' / 'ops' / 'run_newslot1_fill_rate_toxicity_search_latest.py'

RNG_SEED = 20260501
START_BANKROLL = 850.0
PERIODS = list(range(1, 31))
WINDOWS = {'180d': 180, '365d': 365}
TRAIN_DAYS = {'3y': 1095, '5y': 1825, 'full': None}
ENGINES = [x.strip() for x in os.environ.get('TOP159_MULTIDAY_ENGINES', 'lightgbm,catboost,xgboost,logistic').split(',') if x.strip()]
OPTUNA_TRIALS = int(os.environ.get('TOP159_MULTIDAY_OPTUNA_TRIALS', '120'))
MC_TRIALS = int(os.environ.get('TOP159_MULTIDAY_MC_TRIALS', '800'))
MAX_TRAIN_ROWS = int(os.environ.get('TOP159_MULTIDAY_MAX_TRAIN_ROWS', '0'))
FULL_EVAL_LIMIT = int(os.environ.get('TOP159_MULTIDAY_FULL_EVAL_LIMIT', '900'))
STRICT_EVAL_LIMIT = int(os.environ.get('TOP159_MULTIDAY_STRICT_EVAL_LIMIT', '600'))
WRITE_ALL_CANDIDATES = os.environ.get('TOP159_MULTIDAY_WRITE_ALL_CANDIDATES', '1') != '0'
META_SET_LIMIT = int(os.environ.get('TOP159_MULTIDAY_META_SET_LIMIT', '3'))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

base = _load_module('crypto_search_multiday_aux', BASE_SCRIPT)
fill = _load_module('fill_search_multiday_aux', FILL_SCRIPT)


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


def load_eth_1h() -> pd.DataFrame:
    path = RAW / 'eth_usdt_1h.parquet'
    df = pd.read_parquet(path)
    if 'date' in df.columns:
        df['dt'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
    else:
        ts = pd.to_numeric(df['timestamp'], errors='coerce')
        # Binance parquet in this project is usually ms. Guard seconds/us.
        unit = 'ms'
        med = float(ts.dropna().median()) if ts.notna().any() else 0.0
        if med < 10_000_000_000:
            unit = 's'
        elif med > 10_000_000_000_000:
            unit = 'us'
        df['dt'] = pd.to_datetime(ts, unit=unit, utc=True, errors='coerce')
    df = df.dropna(subset=['dt','open','high','low','close']).sort_values('dt').reset_index(drop=True)
    return df


def daily_closed_bars() -> pd.DataFrame:
    raw = load_eth_1h().copy()
    raw['dt'] = pd.to_datetime(raw['dt'], utc=True)
    d = raw.set_index('dt').resample('1D', label='left', closed='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna().reset_index()
    # Timestamp each row at the moment that daily candle is fully known: next UTC midnight.
    d['dt'] = d['dt'] + pd.Timedelta(days=1)
    now_floor = pd.Timestamp(datetime.now(timezone.utc)).floor('1D')
    d = d[d['dt'] <= now_floor].sort_values('dt').reset_index(drop=True)
    return d


def build_daily_period_features(period_days: int) -> tuple[pd.DataFrame, list[str]]:
    df = daily_closed_bars().copy()
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    vol = df['volume'].astype(float)
    ret = close.pct_change()
    df['label_up'] = (close.shift(-period_days) > close).astype(float)
    # Features are known at row dt because dt is after the candle close.
    for n in [1,2,3,5,7,10,14,21,30,45,60,90,120,180]:
        df[f'dret_{n}'] = close.pct_change(n)
        df[f'dvol_{n}'] = ret.rolling(n).std()
        df[f'drange_{n}'] = (high.rolling(n).max() - low.rolling(n).min()) / close
        df[f'dvolume_z_{n}'] = (vol - vol.rolling(n).mean()) / (vol.rolling(n).std() + 1e-12)
    for span in [3,5,8,13,21,34,55,89,144]:
        ema = close.ewm(span=span, adjust=False).mean()
        df[f'dema_dist_{span}'] = close / ema - 1.0
    for fast, slow in [(3,8),(5,13),(8,21),(13,34),(21,55),(34,89)]:
        ef = close.ewm(span=fast, adjust=False).mean()
        es = close.ewm(span=slow, adjust=False).mean()
        df[f'dema_{fast}_{slow}'] = ef / es - 1.0
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['drsi_14'] = (100 - 100 / (1 + gain / (loss + 1e-12))) / 100.0
    df['dbb_pos_20'] = (close - close.rolling(20).mean()) / (2 * close.rolling(20).std() + 1e-12)
    df['period_days'] = period_days
    df['month_sin'] = np.sin(2 * np.pi * df['dt'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['dt'].dt.month / 12)
    features = [c for c in df.columns if c.startswith(('dret_','dvol_','drange_','dvolume_z_','dema_')) or c in {'drsi_14','dbb_pos_20','month_sin','month_cos'}]
    usable = []
    for c in features:
        s = pd.to_numeric(df[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
        if s.notna().sum() >= max(120, int(len(df) * 0.55)):
            usable.append(c)
    df = df.dropna(subset=usable + ['label_up']).reset_index(drop=True)
    df['label_up'] = df['label_up'].astype(int)
    return df, usable


def top159_params() -> dict[str, Any]:
    report = REPORTS / 'newslot1_fak_execution_loop_latest.json'
    if report.exists():
        try:
            data = json.loads(report.read_text())
            params = data.get('uniqueVerdict', {}).get('selectedParams')
            if isinstance(params, dict):
                return params
        except Exception:
            pass
    return {
        'engine': 'lightgbm', 'train_window': '5y', 'feature_mode': 'trend', 'edge': 0.045,
        'vol_q': 0.999, 'trend_mode': 'none', 'bb_abs_max': 2.0, 'loss_n': 0, 'skip_k': 4,
        'n_estimators': 200, 'learning_rate': 0.02193585345721919, 'reg_lambda': 0.0907758387860903,
        'subsample': 0.926502583070262, 'colsample_bytree': 0.8764270068535669,
        'num_leaves': 36, 'min_child_samples': 80, 'depth': 3,
    }


def top159_candidates(window: str, params: dict[str, Any], start_override: pd.Timestamp | None = None, end_override: pd.Timestamp | None = None) -> pd.DataFrame:
    raw = base.load_raw('ETH', '15m')
    df, features = base.build_features(raw, '15m')
    end = df['dt'].max() if end_override is None else pd.Timestamp(end_override)
    start = end - pd.Timedelta(days=WINDOWS[window]) if start_override is None else pd.Timestamp(start_override)
    val = df[(df['dt'] >= start) & (df['dt'] <= end)].copy()
    train_days = 1825 if params.get('train_window') == '5y' else fill.TRAIN_DAYS.get(params.get('train_window'), 1825) or 3650
    train = df[(df['dt'] < start) & (df['dt'] >= start - pd.Timedelta(days=train_days))].copy()
    if fill.MAX_TRAIN_ROWS > 0 and len(train) > fill.MAX_TRAIN_ROWS:
        train = train.sort_values('dt').iloc[-fill.MAX_TRAIN_ROWS:].copy()
    feats = fill.feature_subset(features, params['feature_mode'])
    model = fill.fit_model(params['engine'], train, feats, params)
    if model is None:
        raise RuntimeError('top159 model failed')
    prob = fill.predict(model, val, feats)
    score = np.maximum(prob, 1.0 - prob)
    pred_up = prob >= 0.5
    mask = score >= 0.5 + float(params['edge'])
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
        'score15': score[mask][keep],
        'pred_up15': pred_up[mask][keep].astype(bool),
        'label_up': selected['label_up'].astype(bool).to_numpy(),
    })
    out['won'] = out['pred_up15'].to_numpy() == out['label_up'].to_numpy()
    out['market_slug'] = out['dt'].map(lambda x: f"eth-updown-15m-{int(pd.Timestamp(x).timestamp())}")
    return out.sort_values('dt').reset_index(drop=True)


def feature_subset(features: list[str], mode: str) -> list[str]:
    if mode == 'compact':
        keep = ['dret_1','dret_2','dret_3','dret_5','dret_7','dvol_7','dvol_14','drange_7','drange_14','dema_3_8','dema_5_13','dema_8_21','drsi_14','dbb_pos_20']
    elif mode == 'trend':
        keep = ['dret_3','dret_5','dret_7','dret_10','dret_14','dret_21','dema_3_8','dema_5_13','dema_8_21','dema_13_34','dema_21_55','dema_dist_13','dema_dist_21','dema_dist_34','dvol_14','drange_14','drsi_14','dbb_pos_20']
    else:
        keep = features
    return [c for c in keep if c in features]


def split_train_val(df: pd.DataFrame, window: str, train_window: str):
    end = df['dt'].max()
    start = end - pd.Timedelta(days=WINDOWS[window])
    val = df[df['dt'] >= start].copy()
    if train_window == 'full':
        train = df[df['dt'] < start].copy()
    else:
        train = df[(df['dt'] < start) & (df['dt'] >= start - pd.Timedelta(days=TRAIN_DAYS[train_window]))].copy()
    if MAX_TRAIN_ROWS > 0 and len(train) > MAX_TRAIN_ROWS:
        train = train.sort_values('dt').iloc[-MAX_TRAIN_ROWS:].copy()
    return train, val


def fit_model(engine: str, train: pd.DataFrame, features: list[str], params: dict[str, Any], random_labels: bool = False):
    x = train[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train['label_up'].astype(int).to_numpy()
    if random_labels:
        rng = np.random.default_rng(RNG_SEED + len(train) + len(features))
        y = rng.permutation(y)
    if len(np.unique(y)) < 2 or len(train) < 300:
        return None
    if engine == 'lightgbm':
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=int(params['n_estimators']), learning_rate=float(params['learning_rate']),
            num_leaves=int(params['num_leaves']), min_child_samples=int(params['min_child_samples']),
            subsample=float(params['subsample']), colsample_bytree=float(params['colsample_bytree']),
            reg_lambda=float(params['reg_lambda']), random_state=RNG_SEED, n_jobs=-1, verbose=-1,
        )
    elif engine == 'catboost':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=int(params['n_estimators']), depth=int(params['depth']), learning_rate=float(params['learning_rate']),
            l2_leaf_reg=float(params['reg_lambda']), loss_function='Logloss', eval_metric='Logloss',
            random_seed=RNG_SEED, verbose=False, thread_count=-1,
        )
    elif engine == 'xgboost':
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=int(params['n_estimators']), max_depth=int(params['depth']), learning_rate=float(params['learning_rate']),
            subsample=float(params['subsample']), colsample_bytree=float(params['colsample_bytree']), reg_lambda=float(params['reg_lambda']),
            random_state=RNG_SEED, n_jobs=-1, eval_metric='logloss', verbosity=0,
        )
    elif engine == 'logistic':
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=1.0, random_state=RNG_SEED))
    else:
        raise ValueError(engine)
    model.fit(x, y)
    return model


def predict(model: Any, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    x = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def ece_score(prob: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    if len(prob) == 0:
        return 0.0
    pred = prob >= 0.5
    conf = np.maximum(prob, 1 - prob)
    ok = pred == labels.astype(bool)
    ece = 0.0
    edges = np.linspace(0.5, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        if mask.any():
            ece += float(mask.mean()) * abs(float(ok[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def param_grid() -> list[dict[str, Any]]:
    rows = []
    engines = [e for e in ENGINES if e in {'lightgbm','catboost','xgboost','logistic'}]
    for engine in engines:
        for train_window in ['3y','5y','full']:
            for feature_mode in ['compact','trend','wide']:
                for n_estimators in ([80,140] if engine != 'logistic' else [1]):
                    rows.append({
                        'engine': engine, 'train_window': train_window, 'feature_mode': feature_mode,
                        'n_estimators': n_estimators, 'learning_rate': 0.035, 'num_leaves': 24,
                        'min_child_samples': 30, 'subsample': 0.9, 'colsample_bytree': 0.9,
                        'reg_lambda': 0.8, 'depth': 4,
                    })
    return rows


def train_best_period(period: int, window: str, frame: pd.DataFrame, features: list[str]):
    best = None; best_pred = None; audits = []
    for params in param_grid():
        feats = feature_subset(features, params['feature_mode'])
        train, val = split_train_val(frame, window, params['train_window'])
        if len(feats) < 6 or len(train) < 500 or len(val) < 40:
            continue
        try:
            model = fit_model(params['engine'], train, feats, params)
            if model is None:
                continue
            prob = predict(model, val, feats)
        except Exception as exc:
            audits.append({'period': period, 'window': window, 'params': params, 'error': str(exc)[:180]})
            continue
        labels = val['label_up'].astype(bool).to_numpy()
        pred = prob >= 0.5
        acc = float((pred == labels).mean())
        conf = float(np.maximum(prob, 1 - prob).mean())
        ece = ece_score(prob, labels)
        # Penalize overconfidence and poor calibration. Accuracy remains primary.
        score = acc - 0.25 * ece - 0.03 * max(0.0, conf - 0.62)
        audit = {'period': period, 'window': window, 'params': params, 'trainRows': int(len(train)), 'validationRows': int(len(val)), 'featureCount': int(len(feats)), 'accuracyPct': round(acc*100, 6), 'avgConfidence': round(conf, 6), 'ece': round(ece, 6), 'score': round(score, 8)}
        audits.append(audit)
        if best is None or score > best['scoreRaw']:
            best = {**audit, 'scoreRaw': score, 'searchCandidates': len(audits)}
            best_pred = val[['dt','label_up']].copy()
            best_pred['prob_up'] = prob
            best_pred['pred_up'] = pred
    if best is None or best_pred is None:
        raise RuntimeError(f'no model for {period}d {window}')
    best['searchCandidates'] = len(audits)
    best['candidateAuditsSample'] = sorted(audits, key=lambda x: x.get('score', -9), reverse=True)[:8]
    return best, best_pred.sort_values('dt').reset_index(drop=True), audits


def random_label_audit(period: int, window: str, frame: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    params = {'engine': 'lightgbm', 'train_window': '5y', 'feature_mode': 'trend', 'n_estimators': 80, 'learning_rate': 0.035, 'num_leaves': 24, 'min_child_samples': 30, 'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_lambda': 0.8, 'depth': 4}
    feats = feature_subset(features, 'trend')
    train, val = split_train_val(frame, window, '5y')
    if len(train) < 500 or len(val) < 40:
        return {'period': period, 'window': window, 'status': 'insufficient'}
    model = fit_model('lightgbm', train, feats, params, random_labels=True)
    if model is None:
        return {'period': period, 'window': window, 'status': 'fit_failed'}
    prob = predict(model, val, feats)
    labels = val['label_up'].astype(bool).to_numpy()
    acc = float(((prob >= 0.5) == labels).mean())
    return {'period': period, 'window': window, 'status': 'ok', 'randomLabelAccuracyPct': round(acc*100, 6), 'passed': acc < 0.57}


def attach_period(cands: pd.DataFrame, pred: pd.DataFrame, period: int) -> pd.DataFrame:
    left = cands.sort_values('dt').copy()
    right = pred[['dt','prob_up']].sort_values('dt').copy().rename(columns={'prob_up': f'p_{period}d_up'})
    # pandas merge_asof requires identical datetime precision.  Different parquet
    # readers can yield ms vs ns timestamps even when both are UTC-aware.
    left['dt'] = pd.to_datetime(left['dt'], utc=True).astype('datetime64[ns, UTC]')
    right['dt'] = pd.to_datetime(right['dt'], utc=True).astype('datetime64[ns, UTC]')
    merged = pd.merge_asof(left, right, on='dt', direction='backward', tolerance=pd.Timedelta(days=2))
    p = pd.to_numeric(merged[f'p_{period}d_up'], errors='coerce')
    merged[f'{period}d_available'] = p.notna().astype(int)
    merged[f'p_{period}d_same'] = np.where(merged['pred_up15'], p, 1 - p)
    merged[f'p_{period}d_same'] = merged[f'p_{period}d_same'].fillna(0.5)
    return merged


def add_aux(cands: pd.DataFrame, preds: dict[int, pd.DataFrame], allowed: set[int]) -> pd.DataFrame:
    out = cands.copy()
    for p, pred in preds.items():
        out = attach_period(out, pred, p)
        if p not in allowed:
            out[f'{p}d_available'] = 0
            out[f'p_{p}d_same'] = 0.5
    return out


def curve_metrics(df: pd.DataFrame, name: str, window: str, method: str) -> dict[str, Any]:
    won = df['won'].astype(bool).to_numpy()
    dt = pd.to_datetime(df['dt'], utc=True)
    returns50 = base.returns_from_wins(won, 0.50)
    fill_expect = np.where(won, base.SLOT1_WIN_FILL, base.SLOT1_LOSS_FILL)
    det = base.summarize_curve(dt.reset_index(drop=True), won, returns50, fill_expect, method + '_slot1_toxic_expect', 'ETH', '15m', window, name, {})
    old_trials = base.TRIALS
    base.TRIALS = MC_TRIALS
    try:
        mc = base.monte_carlo_slot1(dt.reset_index(drop=True), won, returns50, RNG_SEED + (180 if window == '180d' else 365) + len(df))
    finally:
        base.TRIALS = old_trials
    fak2 = base.summarize_curve(dt.reset_index(drop=True), won, base.returns_from_wins(won, 0.52), np.ones(len(won)), method + '_fak_ask_plus_2c', 'ETH', '15m', window, name, {})
    return {
        'name': name, 'window': window, 'method': method,
        'trades': int(len(df)), 'wins': int(won.sum()), 'losses': int((~won).sum()),
        'winRatePct': round(float(won.mean()) * 100, 6) if len(won) else 0.0,
        'endingBankrollToxicExpected': det['endingBankroll'],
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


def evaluate_variant(base_df: pd.DataFrame, kept: pd.DataFrame, name: str, window: str, method: str) -> dict[str, Any]:
    row = curve_metrics(kept, name, window, method)
    base_idx = set(base_df.index)
    keep_idx = set(kept.index)
    blocked = base_df.loc[list(base_idx - keep_idx)] if base_idx - keep_idx else base_df.iloc[0:0]
    row.update({
        'baseTrades': int(len(base_df)),
        'blockedTrades': int(len(blocked)),
        'blockedWinners': int(blocked['won'].sum()) if len(blocked) else 0,
        'blockedLosers': int((~blocked['won']).sum()) if len(blocked) else 0,
        'retentionPct': round(100.0 * len(kept) / len(base_df), 6) if len(base_df) else 0.0,
        'blockedSetHash': stable_hash(list(blocked['dt'].astype(str).head(800)) + [len(blocked), int(blocked['won'].sum())]) if len(blocked) else 'empty',
    })
    return row


def weighted_filter(df: pd.DataFrame, periods: list[int], weights: dict[int, float], amp: float, neg_clip: float, pos_clip: float, threshold: float) -> pd.DataFrame:
    support = np.zeros(len(df), dtype=float)
    denom = 0.0
    for p in periods:
        w = float(weights.get(p, 0.0))
        if abs(w) <= 1e-12:
            continue
        avail = df[f'{p}d_available'].to_numpy(dtype=float)
        same = df[f'p_{p}d_same'].to_numpy(dtype=float)
        support += w * avail * (2 * same - 1)
        denom += abs(w)
    if denom > 0:
        support = support / denom
        final = df['score15'].to_numpy(dtype=float) + np.clip(amp * support, -neg_clip, pos_clip)
    else:
        support[:] = 0.0
        final = df['score15'].to_numpy(dtype=float)
    out = df.copy()
    out['aux_support'] = support
    out['final_score'] = final
    return out[out['final_score'] >= threshold].copy()


def hard_veto_filter(df: pd.DataFrame, periods: list[int], oppose_same_threshold: float, min_oppose: int) -> pd.DataFrame:
    oppose = np.zeros(len(df), dtype=int)
    for p in periods:
        same = df[f'p_{p}d_same'].to_numpy(dtype=float)
        avail = df[f'{p}d_available'].to_numpy(dtype=bool)
        oppose += (same <= oppose_same_threshold) & avail
    out = df.copy(); out['oppose_count'] = oppose
    return out[out['oppose_count'] < min_oppose].copy()


def meta_filter(train_df: pd.DataFrame, val_df: pd.DataFrame, periods: list[int], engine: str, threshold: float):
    feats = ['score15'] + [f'p_{p}d_same' for p in periods] + [f'{p}d_available' for p in periods]
    train = train_df.dropna(subset=feats + ['won']).copy()
    if len(train) < 250 or train['won'].nunique() < 2:
        return val_df.iloc[0:0].copy(), {'status': 'insufficient', 'trainRows': len(train)}
    params = {'engine': engine, 'train_window': 'meta', 'feature_mode': 'meta', 'n_estimators': 120, 'learning_rate': 0.04, 'num_leaves': 16, 'min_child_samples': 40, 'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_lambda': 1.0, 'depth': 3}
    # The candidate frame already contains the original top159 label columns.
    # Build a clean training frame so the meta model has exactly one target.
    train2 = train[feats].copy()
    train2['label_up'] = train['won'].astype(int).to_numpy()
    model = fit_model(engine, train2, feats, params)
    if model is None:
        return val_df.iloc[0:0].copy(), {'status': 'fit_failed'}
    prob = predict(model, val_df, feats)
    out = val_df.copy(); out['meta_prob'] = prob
    return out[out['meta_prob'] >= threshold].copy(), {'status': 'ok', 'trainRows': len(train), 'threshold': threshold, 'engine': engine}


def build_meta_train(val_start: pd.Timestamp, top_params: dict[str, Any], preds_by_period: dict[int, pd.DataFrame], allowed: set[int]) -> pd.DataFrame:
    start = val_start - pd.Timedelta(days=365)
    end = val_start - pd.Timedelta(minutes=15)
    cands = top159_candidates('365d', top_params, start_override=start, end_override=end)
    return add_aux(cands, preds_by_period, allowed)


def pass_gate(row180: dict[str, Any], row365: dict[str, Any], base180: dict[str, Any], base365: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    if row180['trades'] < 100:
        reasons.append('180d_trades_below_100')
    if row365['trades'] < 200:
        reasons.append('365d_trades_below_200')
    if row365['retentionPct'] < 45:
        reasons.append('365d_retention_below_45pct')
    if row180['fak2EndingBankroll'] <= 850 or row365['fak2EndingBankroll'] <= 850:
        reasons.append('fak2_not_profitable')
    if row180['endingBankrollP50'] < base180['endingBankrollP50'] * 0.995:
        reasons.append('180d_toxic_p50_worse')
    if row365['endingBankrollP50'] < base365['endingBankrollP50'] * 0.995:
        reasons.append('365d_toxic_p50_worse')
    if row180['blockedLosers'] <= row180['blockedWinners']:
        reasons.append('180d_blocked_losers_not_more_than_winners')
    if row365['blockedLosers'] <= row365['blockedWinners']:
        reasons.append('365d_blocked_losers_not_more_than_winners')
    return len(reasons) == 0, reasons


def top_periods(model_truth: dict[str, Any], top_n: int) -> list[int]:
    rows = []
    for p, win_map in model_truth.items():
        if p == 'bugcheck':
            continue
        p_int = int(p.replace('d',''))
        a180 = win_map['180d']['accuracyPct']
        a365 = win_map['365d']['accuracyPct']
        allowed = win_map['allowedForLiveWeight']
        if allowed:
            rows.append((0.5 * (a180 + a365), p_int))
    rows.sort(reverse=True)
    return [p for _, p in rows[:top_n]]


def all_period_combos(allowed: list[int], max_len: int = 4) -> tuple[list[tuple[int, ...]], dict[str, Any]]:
    allowed_set = set(allowed)
    combos: set[tuple[int, ...]] = set()
    for r in range(1, max_len + 1):
        combos.update(tuple(x) for x in itertools.combinations(sorted(allowed_set), r))
    forced = [(4,), (4, 7), (7, 11), (4, 11, 18), (4, 7, 11, 18)]
    forced_status = {}
    for f in forced:
        ok = all(p in allowed_set for p in f)
        forced_status['/'.join(map(str, f))] = 'evaluated' if ok else 'skipped_period_not_allowed'
        if ok:
            combos.add(tuple(f))
    return sorted(combos, key=lambda x: (len(x), x)), forced_status


def aux_score_array(df: pd.DataFrame, periods: tuple[int, ...], mode: str) -> np.ndarray:
    vals = []
    midpoint = float(np.median(periods)) if periods else 0.0
    for p in periods:
        same = df[f'p_{p}d_same'].to_numpy(dtype=float)
        avail = df[f'{p}d_available'].to_numpy(dtype=bool)
        if mode == 'same':
            v = same
        elif mode == 'reverse':
            v = 1.0 - same
        elif mode == 'short_same_long_reverse':
            v = same if p <= midpoint else 1.0 - same
        elif mode == 'short_reverse_long_same':
            v = 1.0 - same if p <= midpoint else same
        else:
            raise ValueError(mode)
        vals.append(np.where(avail, v, 0.5))
    if not vals:
        return np.full(len(df), 0.5, dtype=float)
    return np.vstack(vals).mean(axis=0)


def apply_candidate_mask(df: pd.DataFrame, kind: str, params: dict[str, Any]) -> np.ndarray:
    periods = tuple(int(p) for p in params.get('periods', []))
    mode = params.get('mode', 'same')
    score = aux_score_array(df, periods, mode)
    if kind == 'avg_low_block':
        keep = score >= float(params['threshold'])
    elif kind == 'avg_high_block':
        keep = score <= float(params['threshold'])
    elif kind == 'count_low_block':
        cnt = np.zeros(len(df), dtype=int)
        for p in periods:
            same = df[f'p_{p}d_same'].to_numpy(dtype=float)
            if mode == 'reverse':
                same = 1.0 - same
            cnt += same <= float(params['threshold'])
        keep = cnt < int(params['minCount'])
    elif kind == 'count_high_block':
        cnt = np.zeros(len(df), dtype=int)
        for p in periods:
            same = df[f'p_{p}d_same'].to_numpy(dtype=float)
            if mode == 'reverse':
                same = 1.0 - same
            cnt += same >= float(params['threshold'])
        keep = cnt < int(params['minCount'])
    elif kind == 'segmented_low_score_block':
        raw_score = df['score15'].to_numpy(dtype=float)
        keep = (raw_score >= float(params['scoreCutoff'])) | (score >= float(params['threshold']))
    elif kind == 'direction_specific_low_block':
        pred_up = df['pred_up15'].to_numpy(dtype=bool)
        th = np.where(pred_up, float(params['thresholdUp']), float(params['thresholdDown']))
        keep = score >= th
    elif kind == 'weighted_soft':
        weights = {int(k): float(v) for k, v in params['weights'].items()}
        kept = weighted_filter(
            df,
            list(periods),
            weights,
            float(params['amp']),
            float(params['negClip']),
            float(params['posClip']),
            float(params['threshold']),
        )
        keep_idx = set(kept.index)
        keep = np.array([i in keep_idx for i in df.index], dtype=bool)
    elif kind == 'hard_veto':
        kept = hard_veto_filter(df, list(periods), float(params['opposeSameThreshold']), int(params['minOppose']))
        keep_idx = set(kept.index)
        keep = np.array([i in keep_idx for i in df.index], dtype=bool)
    else:
        raise ValueError(f'unknown candidate kind {kind}')
    return np.asarray(keep, dtype=bool)


def quick_metrics(base_df: pd.DataFrame, keep_mask: np.ndarray, name: str, window: str, kind: str) -> dict[str, Any]:
    won = base_df['won'].astype(bool).to_numpy()
    keep_mask = np.asarray(keep_mask, dtype=bool)
    kept = won[keep_mask]
    blocked = won[~keep_mask]
    return {
        'name': name, 'window': window, 'kind': kind,
        'trades': int(keep_mask.sum()),
        'wins': int(kept.sum()),
        'losses': int((~kept).sum()),
        'winRatePct': round(float(kept.mean()) * 100, 6) if len(kept) else 0.0,
        'baseTrades': int(len(base_df)),
        'blockedTrades': int((~keep_mask).sum()),
        'blockedWinners': int(blocked.sum()) if len(blocked) else 0,
        'blockedLosers': int((~blocked).sum()) if len(blocked) else 0,
        'retentionPct': round(100.0 * float(keep_mask.mean()), 6) if len(base_df) else 0.0,
    }


def quick_candidate_score(row180: dict[str, Any], row365: dict[str, Any]) -> float:
    d180 = row180['blockedLosers'] - row180['blockedWinners']
    d365 = row365['blockedLosers'] - row365['blockedWinners']
    trade_penalty = 0.0
    if row180['trades'] < 100:
        trade_penalty += (100 - row180['trades']) * 10.0
    if row365['trades'] < 200:
        trade_penalty += (200 - row365['trades']) * 8.0
    if row365['retentionPct'] < 45.0:
        trade_penalty += (45.0 - row365['retentionPct']) * 30.0
    return (
        min(d180, d365) * 10000.0
        + (d180 + d365) * 120.0
        + (row180['winRatePct'] + row365['winRatePct']) * 10.0
        + row365['retentionPct']
        - trade_penalty
    )


def summarize_quick_candidate(kind: str, params: dict[str, Any], windows_df: dict[str, pd.DataFrame]) -> dict[str, Any]:
    candidate_id = stable_hash([kind, json.dumps(params, sort_keys=True)])
    name = f'{kind}_{candidate_id}'
    rows = []
    for window, df in windows_df.items():
        mask = apply_candidate_mask(df, kind, params)
        rows.append(quick_metrics(df, mask, name, window, kind))
    by = {r['window']: r for r in rows}
    row180 = by['180d']; row365 = by['365d']
    structure_ok = (
        row180['blockedLosers'] > row180['blockedWinners']
        and row365['blockedLosers'] > row365['blockedWinners']
        and row180['trades'] >= 100
        and row365['trades'] >= 200
        and row365['retentionPct'] >= 45.0
    )
    return {
        'name': name,
        'kind': kind,
        'params': params,
        'quickRows': rows,
        'structureOk': structure_ok,
        'quickScore': quick_candidate_score(row180, row365),
    }


def full_eval_candidate(summary: dict[str, Any], windows_df: dict[str, pd.DataFrame]) -> dict[str, Any]:
    rows = []
    for window, df in windows_df.items():
        mask = apply_candidate_mask(df, summary['kind'], summary['params'])
        kept = df.loc[mask].copy()
        rows.append(evaluate_variant(df, kept, summary['name'], window, summary['kind']))
    score = (
        rows[0]['endingBankrollP50']
        + rows[1]['endingBankrollP50']
        - rows[1]['maxDrawdownP50']
        + (rows[0]['blockedLosers'] - rows[0]['blockedWinners']) * 30.0
        + (rows[1]['blockedLosers'] - rows[1]['blockedWinners']) * 30.0
    )
    return {'name': summary['name'], 'kind': summary['kind'], 'params': summary['params'], 'rows': rows, 'score': score, 'quickRows': summary['quickRows'], 'structureOk': summary['structureOk']}


def candidate_param_stream(combos: list[tuple[int, ...]], model_truth: dict[str, Any]):
    modes = ['same', 'reverse', 'short_same_long_reverse', 'short_reverse_long_same']
    forced = {(4,), (4, 7), (7, 11), (4, 11, 18), (4, 7, 11, 18)}
    for ps in combos:
        # Cover every 1-4 period combination, but keep the universal pass compact.
        # Extra thresholds are reserved for the exact user-requested combinations
        # and the Optuna/refinement stage; otherwise the Cartesian product becomes
        # millions of mostly redundant candidates.
        rich = ps in forced or len(ps) == 1
        for mode in modes:
            low_thresholds = [0.46, 0.50] + ([0.40, 0.43, 0.52, 0.55] if rich else [])
            high_thresholds = [0.58] + ([0.55, 0.61, 0.64] if rich else [])
            for th in low_thresholds:
                yield 'avg_low_block', {'periods': list(ps), 'mode': mode, 'threshold': th}
            for th in high_thresholds:
                yield 'avg_high_block', {'periods': list(ps), 'mode': mode, 'threshold': th}
            count_thresholds = [0.46] + ([0.43, 0.49] if rich else [])
            for th in count_thresholds:
                for min_count in range(1, min((3 if rich else 1), len(ps)) + 1):
                    yield 'count_low_block', {'periods': list(ps), 'mode': mode, 'threshold': th, 'minCount': min_count}
            for th in ([0.60] + ([0.56, 0.64] if rich else [])):
                for min_count in range(1, min((3 if rich else 1), len(ps)) + 1):
                    yield 'count_high_block', {'periods': list(ps), 'mode': mode, 'threshold': th, 'minCount': min_count}
            for score_cut in ([0.565] + ([0.555, 0.575] if rich else [])):
                for th in ([0.46] + ([0.43, 0.49, 0.52] if rich else [])):
                    yield 'segmented_low_score_block', {'periods': list(ps), 'mode': mode, 'scoreCutoff': score_cut, 'threshold': th}
            ds_pairs = [(0.46, 0.46)]
            if rich:
                ds_pairs += [(0.43, 0.43), (0.46, 0.43), (0.43, 0.46), (0.49, 0.46), (0.46, 0.49), (0.52, 0.49)]
            for up_th, dn_th in ds_pairs:
                yield 'direction_specific_low_block', {'periods': list(ps), 'mode': mode, 'thresholdUp': up_th, 'thresholdDown': dn_th}


def run_combo_search(windows_df: dict[str, pd.DataFrame], model_truth: dict[str, Any], base_rows: dict[str, dict[str, Any]]):
    allowed_all = top_periods(model_truth, 30)
    combos, forced_status = all_period_combos(allowed_all, max_len=4)
    summary_path = REPORTS / 'top159_multiday_aux_all_candidate_summary_latest.jsonl'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    total_seen = 0
    with summary_path.open('w', encoding='utf-8') as fh:
        for kind, params in candidate_param_stream(combos, model_truth):
            total_seen += 1
            s = summarize_quick_candidate(kind, params, windows_df)
            summaries.append(s)
            if WRITE_ALL_CANDIDATES:
                compact = {
                    'name': s['name'],
                    'kind': s['kind'],
                    'params': s['params'],
                    'structureOk': s['structureOk'],
                    'quickScore': round(float(s['quickScore']), 6),
                    'quickRows': s['quickRows'],
                }
                fh.write(json.dumps(compact, ensure_ascii=False, separators=(',', ':')) + '\n')
            if total_seen % 50000 == 0:
                print(f'[top159-multiday] coarse candidates evaluated: {total_seen}', flush=True)

    summaries.sort(key=lambda x: x['quickScore'], reverse=True)
    strict = [s for s in summaries if s['structureOk']]
    selected_summaries: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for s in strict[:STRICT_EVAL_LIMIT] + summaries[:FULL_EVAL_LIMIT]:
        if s['name'] not in seen_names:
            selected_summaries.append(s)
            seen_names.add(s['name'])

    # Always fully evaluate the exact examples the user requested if they exist.
    requested_sets = [(4,), (4, 7), (7, 11), (4, 11, 18), (4, 7, 11, 18)]
    requested_keys = {tuple(x) for x in requested_sets if all(p in allowed_all for p in x)}
    for s in summaries:
        if tuple(s['params'].get('periods', [])) in requested_keys and s['name'] not in seen_names:
            selected_summaries.append(s)
            seen_names.add(s['name'])
            if len([x for x in selected_summaries if tuple(x['params'].get('periods', [])) in requested_keys]) >= 250:
                break

    candidates = []
    for i, s in enumerate(selected_summaries, 1):
        candidates.append(full_eval_candidate(s, windows_df))
        if i % 100 == 0:
            print(f'[top159-multiday] full candidates evaluated: {i}/{len(selected_summaries)}', flush=True)

    # Optuna for soft weighted gate across top 12 periods.  It is retained as a
    # second-stage math search, but strict pass/fail still goes through pass_gate.
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        ps_opt = top_periods(model_truth, 12)
        if ps_opt:
            def objective(trial):
                raw = {p: trial.suggest_float(f'w_{p}', -1.0, 1.0) for p in ps_opt}
                s = sum(abs(v) for v in raw.values()) or 1.0
                weights = {p: v/s for p, v in raw.items()}
                amp = trial.suggest_float('amp', 0.005, 0.18)
                neg = trial.suggest_float('neg', 0.01, 0.18)
                pos = trial.suggest_float('pos', 0.0, 0.12)
                th = trial.suggest_float('threshold', 0.535, 0.59)
                vals = []
                for window, df in windows_df.items():
                    kept = weighted_filter(df, ps_opt, weights, amp, neg, pos, th)
                    vals.append(evaluate_variant(df, kept, f'optuna_trial_{trial.number}', window, 'weighted_multiday_aux_optuna'))
                r180, r365 = vals
                toxicity = (r180['blockedLosers'] - r180['blockedWinners']) + (r365['blockedLosers'] - r365['blockedWinners'])
                toxicity_floor = min(r180['blockedLosers'] - r180['blockedWinners'], r365['blockedLosers'] - r365['blockedWinners'])
                return float(r365['endingBankrollP50']), float(r180['endingBankrollP50']), float(r365['maxDrawdownP50']), float(toxicity + 10.0 * toxicity_floor)
            study = optuna.create_study(directions=['maximize','maximize','minimize','maximize'], sampler=optuna.samplers.NSGAIISampler(seed=RNG_SEED))
            study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False, gc_after_trial=True)
            for t in study.best_trials[:48]:
                raw = {p: t.params.get(f'w_{p}', 0.0) for p in ps_opt}
                norm = sum(abs(v) for v in raw.values()) or 1.0
                weights = {p: v / norm for p, v in raw.items()}
                params = {'periods': ps_opt, 'weights': weights, 'amp': t.params['amp'], 'negClip': t.params['neg'], 'posClip': t.params['pos'], 'threshold': t.params['threshold'], 'optunaTrial': t.number, 'values': t.values}
                name = f"weighted_optuna_{stable_hash([t.number, params])}"
                rows = []
                for window, df in windows_df.items():
                    kept = weighted_filter(df, ps_opt, weights, t.params['amp'], t.params['neg'], t.params['pos'], t.params['threshold'])
                    rows.append(evaluate_variant(df, kept, name, window, 'weighted_multiday_aux_optuna'))
                score = rows[0]['endingBankrollP50'] + rows[1]['endingBankrollP50'] - rows[1]['maxDrawdownP50'] + (rows[0]['blockedLosers'] - rows[0]['blockedWinners'] + rows[1]['blockedLosers'] - rows[1]['blockedWinners']) * 30.0
                candidates.append({'name': name, 'kind': 'weighted_optuna', 'params': params, 'rows': rows, 'score': score})
    except Exception as exc:
        candidates.append({'name': 'optuna_unavailable', 'kind': 'diagnostic', 'params': {'error': str(exc)[:240]}, 'rows': [], 'score': -1e18})

    # Meta gate using representative period sets; this is expensive, so keep it
    # targeted after the exhaustive rule sweep above.
    representative_sets = []
    if META_SET_LIMIT > 0:
        for s in summaries:
            ps = tuple(s['params'].get('periods', []))
            if ps and ps not in representative_sets:
                representative_sets.append(ps)
            if len(representative_sets) >= META_SET_LIMIT:
                break
    meta_train_cache: dict[str, pd.DataFrame] = {}
    for ps in representative_sets:
        for engine in ['lightgbm','logistic']:
            for th in [0.50,0.54,0.58]:
                rows = []; ok = True; audits = []
                for window, df in windows_df.items():
                    if window not in meta_train_cache:
                        val_start = pd.to_datetime(df['dt'], utc=True).min()
                        meta_train_cache[window] = build_meta_train(val_start, top159_params(), {p: preds_by_window[window][p] for p in preds_by_window[window]}, set(allowed_periods))
                    train_df = meta_train_cache[window]
                    kept, ma = meta_filter(train_df, df, list(ps), engine, th)
                    audits.append({'window': window, **ma})
                    if len(kept) == 0:
                        ok = False; break
                    rows.append(evaluate_variant(df, kept, f'meta_{engine}_{stable_hash([ps,th])}', window, 'meta_multiday_aux'))
                if ok:
                    score = rows[0]['endingBankrollP50'] + rows[1]['endingBankrollP50'] - rows[1]['maxDrawdownP50'] + (rows[0]['blockedLosers'] - rows[0]['blockedWinners'] + rows[1]['blockedLosers'] - rows[1]['blockedWinners']) * 30.0
                    candidates.append({'name': rows[0]['name'], 'kind': 'meta', 'params': {'periods': list(ps), 'engine': engine, 'threshold': th, 'audits': audits}, 'rows': rows, 'score': score})

    candidates = [c for c in candidates if c.get('rows')]
    candidates.sort(key=lambda x: x['score'], reverse=True)
    verdicts = []
    for c in candidates:
        by = {r['window']: r for r in c['rows']}
        passed, reasons = pass_gate(by['180d'], by['365d'], base_rows['180d'], base_rows['365d'])
        verdicts.append({'name': c['name'], 'kind': c['kind'], 'params': c['params'], 'passed': passed, 'reasons': reasons, 'rows': c['rows'], 'quickRows': c.get('quickRows')})
    search_audit = {
        'allowedPeriods': allowed_all,
        'comboCount': len(combos),
        'forcedComboStatus': forced_status,
        'coarseCandidatesEvaluated': total_seen,
        'coarseSummaryPath': str(summary_path),
        'structureOkCount': len(strict),
        'fullCandidatesEvaluated': len(candidates),
        'fullEvalLimit': FULL_EVAL_LIMIT,
        'strictEvalLimit': STRICT_EVAL_LIMIT,
    }
    return candidates, verdicts, search_audit, summaries


# Global set by main for meta builder; kept explicit to avoid hidden live state.
preds_by_window: dict[str, dict[int, pd.DataFrame]] = {}
allowed_periods: list[int] = []
base_rows: dict[str, dict[str, Any]] = {}


def main() -> int:
    generated_at = now_iso()
    top_params = top159_params()
    data_truth = {
        'generatedAt': generated_at,
        'source': 'local eth_usdt_1h resampled to closed daily bars; Binance official only has limited native day intervals',
        'raw1hRows': int(len(load_eth_1h())),
        'dailyRows': int(len(daily_closed_bars())),
        'periods': PERIODS,
        'liveMutation': 'none',
    }

    frames: dict[int, tuple[pd.DataFrame, list[str]]] = {}
    model_truth: dict[str, Any] = {'generatedAt': generated_at, 'periods': {}, 'bugcheck': {}}
    global preds_by_window, allowed_periods, base_rows
    preds_by_window = {'180d': {}, '365d': {}}
    all_audits = {}
    for p in PERIODS:
        print(f'[top159-multiday] training {p}d auxiliary models...', flush=True)
        df, feats = build_daily_period_features(p)
        frames[p] = (df, feats)
        model_truth['periods'][f'{p}d'] = {}
        pass_both = True
        for w in ['180d','365d']:
            best, pred, audits = train_best_period(p, w, df, feats)
            all_audits[f'{p}d_{w}'] = audits
            model_truth['periods'][f'{p}d'][w] = {k: v for k, v in best.items() if k != 'scoreRaw'}
            preds_by_window[w][p] = pred
            if best['accuracyPct'] <= 50.0:
                pass_both = False
            print(
                f"[top159-multiday] {p}d {w}: best={best.get('params', {}).get('engine', 'unknown')} "
                f"acc={best['accuracyPct']:.3f}% ece={best.get('ece', 0):.4f}",
                flush=True,
            )
        model_truth['periods'][f'{p}d']['allowedForLiveWeight'] = pass_both
    allowed_periods = [p for p in PERIODS if model_truth['periods'][f'{p}d']['allowedForLiveWeight']]

    # Random-label leakage checks on short, medium and long horizons.
    random_checks = []
    for p in [1, 7, 15, 30]:
        df, feats = frames[p]
        for w in ['180d','365d']:
            random_checks.append(random_label_audit(p, w, df, feats))
    model_truth['bugcheck'] = {
        'futureFieldBlacklist': sorted(list(base.FORBIDDEN_FEATURES)) + list(base.FORBIDDEN_PREFIXES),
        'featureTiming': 'daily rows timestamped after candle close; labels use future closes only as labels',
        'alignment': 'merge_asof backward from 15m candidates to latest closed daily prediction',
        'randomLabelChecks': random_checks,
        'randomLabelPassed': all(x.get('passed', True) for x in random_checks if x.get('status') == 'ok'),
        'windowIsolation': '180d and 365d top159 candidates and aux models are trained/evaluated independently',
    }

    windows_df = {}
    base_rows = {}
    for w in ['180d','365d']:
        cands = top159_candidates(w, top_params)
        cands = add_aux(cands, preds_by_window[w], set(allowed_periods))
        windows_df[w] = cands
        br = curve_metrics(cands, 'current_top159', w, 'baseline')
        br.update({'baseTrades': len(cands), 'blockedTrades': 0, 'blockedWinners': 0, 'blockedLosers': 0, 'retentionPct': 100.0})
        base_rows[w] = br

    candidates, verdicts, search_audit, coarse_summaries = run_combo_search(windows_df, model_truth['periods'], base_rows)
    passed = [v for v in verdicts if v['passed']]
    if passed:
        selected = sorted(passed, key=lambda v: (v['rows'][1]['endingBankrollP50'], -v['rows'][1]['maxDrawdownP50'], v['rows'][1]['blockedLosers'] - v['rows'][1]['blockedWinners']), reverse=True)[0]
        status = 'candidate_passed_research_gate'
        live_action = 'research_only_prepare_preflight_before_any_live_change'
    else:
        selected = None
        status = 'no_candidate_passed_keep_current_top159'
        live_action = 'do_not_enable_aux_gate'

    compare_rows = [base_rows['180d'], base_rows['365d']]
    if selected:
        compare_rows.extend(selected['rows'])
    for c in candidates[:14]:
        if selected and c['name'] == selected['name']:
            continue
        compare_rows.extend(c['rows'])

    payload = {
        'generatedAt': generated_at,
        'status': status,
        'liveAction': live_action,
        'top159Params': top_params,
        'dataTruth': data_truth,
        'modelTruth': model_truth,
        'allowedPeriods': allowed_periods,
        'searchAudit': search_audit,
        'compareRows': compare_rows,
        'candidateVerdicts': verdicts,
        'selected': selected,
    }

    write_json(REPORTS / 'top159_multiday_aux_data_truth_latest.json', data_truth)
    write_json(REPORTS / 'top159_multiday_aux_model_audit_latest.json', model_truth)
    # Compact single-period table.
    single_rows = []
    for p in PERIODS:
        m = model_truth['periods'][f'{p}d']
        single_rows.append({
            'period': f'{p}d',
            'allowedForLiveWeight': m['allowedForLiveWeight'],
            'accuracy180': m['180d']['accuracyPct'],
            'accuracy365': m['365d']['accuracyPct'],
            'engine180': m['180d']['params']['engine'],
            'engine365': m['365d']['params']['engine'],
            'ece180': m['180d']['ece'],
            'ece365': m['365d']['ece'],
            'validationRows180': m['180d']['validationRows'],
            'validationRows365': m['365d']['validationRows'],
        })
    write_json(REPORTS / 'top159_multiday_aux_single_period_scoreboard_latest.json', {'generatedAt': generated_at, 'rows': single_rows})
    write_json(REPORTS / 'top159_multiday_aux_combo_hyperopt_latest.json', {'generatedAt': generated_at, 'searchAudit': search_audit, 'candidates': candidates})
    write_json(REPORTS / 'top159_multiday_aux_coarse_top_latest.json', {'generatedAt': generated_at, 'searchAudit': search_audit, 'topCoarseCandidates': coarse_summaries[:5000]})
    write_json(REPORTS / 'top159_multiday_aux_180_365_pressure_compare_latest.json', {'generatedAt': generated_at, 'rows': compare_rows})
    write_json(REPORTS / 'top159_multiday_aux_unique_verdict_latest.json', {'generatedAt': generated_at, 'status': status, 'selected': selected, 'liveAction': live_action})
    write_json(REPORTS / 'top159_multiday_aux_research_latest.json', payload)

    lines = ['# top159 1d～30d 多日趋势辅助门研究', '', f'生成时间：`{generated_at}`', '', '## 结论', f'- 状态：`{status}`', f'- live动作：`{live_action}`', f'- 允许进入组合的周期：`{allowed_periods}`', f"- 全组合数：`{search_audit['comboCount']}`，粗筛候选：`{search_audit['coarseCandidatesEvaluated']}`，完整复核候选：`{search_audit['fullCandidatesEvaluated']}`", f"- 指定组合覆盖：`{json.dumps(search_audit['forcedComboStatus'], ensure_ascii=False)}`", f"- 全部粗筛摘要：`{search_audit['coarseSummaryPath']}`", '']
    lines += ['## 1d～30d 单模型表', '|周期|允许入组合|180准确率|365准确率|180模型|365模型|180样本|365样本|180校准误差|365校准误差|', '|---|---:|---:|---:|---|---|---:|---:|---:|---:|']
    for r in single_rows:
        lines.append(f"|{r['period']}|{r['allowedForLiveWeight']}|{r['accuracy180']}|{r['accuracy365']}|{r['engine180']}|{r['engine365']}|{r['validationRows180']}|{r['validationRows365']}|{r['ece180']}|{r['ece365']}|")
    lines += ['', '## 组合绝对对比表', '|方法|窗口|交易数|胜/负|胜率|P5资金|P50资金|P95资金|最大回撤P50|月正收益|FAK+2分资金|拦截单|拦截赢家|拦截输家|保留率|哈希|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in compare_rows:
        lines.append(f"|{r['name']} / {r['method']}|{r['window']}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']}|{r['endingBankrollP5']}|{r['endingBankrollP50']}|{r['endingBankrollP95']}|{r['maxDrawdownP50']}|{r['monthlyPositiveRatioP50']}|{r['fak2EndingBankroll']}|{r.get('blockedTrades',0)}|{r.get('blockedWinners',0)}|{r.get('blockedLosers',0)}|{r.get('retentionPct',100)}|`{r['setHash']}`|")
    if selected:
        lines += ['', '## 选中候选', f"- `{selected['name']}`", f"- 参数：`{json.dumps(selected['params'], ensure_ascii=False)}`"]
    else:
        # Show top failure reasons.
        reason_counts = {}
        for v in verdicts:
            for reason in v['reasons']:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        lines += ['', '## 唯一结论', '- 当前搜索族没有候选同时通过交易数、毒性资金、FAK+2分和误杀/拦截门槛。', f'- 结构正确候选数：`{search_audit["structureOkCount"]}`', f'- 主要失败原因统计：`{json.dumps(reason_counts, ensure_ascii=False)}`', '- 不接入 live；当前 top159 保持原样。']
    lines += ['', '## 防泄漏审计', f"- 随机标签测试通过：`{model_truth['bugcheck']['randomLabelPassed']}`", '- 特征只来自已收盘日线；15m 候选只向后匹配最近已收盘日线预测。']
    write_text(REPORTS / 'top159_multiday_aux_research_latest.md', '\n'.join(lines) + '\n')

    print(json.dumps({'status': status, 'selected': selected['name'] if selected else None, 'allowedPeriods': allowed_periods, 'report': str(REPORTS / 'top159_multiday_aux_research_latest.md')}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
