#!/usr/bin/env python3
from __future__ import annotations

# Research-only stacked auxiliary probability integration for top159.
# Does not read/write live config. Does not start trading.

import concurrent.futures as cf
import hashlib
import importlib.util
import itertools
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

THREADS = os.environ.get('TOP159_STACKED_WORKER_THREADS', '2')
for k in ['OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(k, THREADS)

import numpy as np
import pandas as pd

ROOT = Path('/Users/mac/polyfun')
REPORTS = ROOT / 'reports'
RAW = ROOT / 'data' / 'raw'
CACHE_DIR = ROOT / 'data' / 'processed' / 'top159_stacked_aux'
BASE_SCRIPT = ROOT / 'scripts' / 'ops' / 'run_crypto15m_1h_multasset_pressure_search_latest.py'
FILL_SCRIPT = ROOT / 'scripts' / 'ops' / 'run_newslot1_fill_rate_toxicity_search_latest.py'
AUX_SCRIPT = ROOT / 'polyfun-next' / 'scripts' / 'run_top159_multiday_aux_research.py'
INTEGRATED_SCRIPT = ROOT / 'polyfun-next' / 'scripts' / 'run_top159_integrated_main_model_24h_search.py'

WORKERS = int(os.environ.get('TOP159_STACKED_WORKERS', '4'))
MAX_SECONDS = int(os.environ.get('TOP159_STACKED_MAX_SECONDS', str(24 * 3600)))
CHECKPOINT_SECONDS = int(os.environ.get('TOP159_STACKED_CHECKPOINT_SECONDS', '900'))
MC_TRIALS = int(os.environ.get('TOP159_STACKED_MC_TRIALS', '800'))
MAX_TRAIN_ROWS = int(os.environ.get('TOP159_STACKED_MAX_TRAIN_ROWS', '180000'))
ENGINES = [x.strip() for x in os.environ.get('TOP159_STACKED_ENGINES', 'lightgbm,xgboost,logistic,catboost').split(',') if x.strip()]
PARAM_LIMIT = int(os.environ.get('TOP159_STACKED_PARAM_LIMIT', '0'))
OOF_BLOCK_DAYS = int(os.environ.get('TOP159_STACKED_OOF_BLOCK_DAYS', '180'))
OOF_TRAIN_DAYS = int(os.environ.get('TOP159_STACKED_OOF_TRAIN_DAYS', '1825'))
MAX_FEATURES = int(os.environ.get('TOP159_STACKED_MAX_FEATURES', '140'))
RNG_SEED = 20260502

AUX_KEYS = ['1h', '4h', '4d', '7d', '18d']

OUT_RESULTS = REPORTS / 'top159_stacked_aux_integrated_results_latest.jsonl'
OUT_CHECKPOINT = REPORTS / 'top159_stacked_aux_integrated_checkpoint_latest.json'
OUT_LEADERBOARD_JSON = REPORTS / 'top159_stacked_aux_integrated_leaderboard_latest.json'
OUT_LEADERBOARD_MD = REPORTS / 'top159_stacked_aux_integrated_leaderboard_latest.md'
OUT_AUDIT_JSON = REPORTS / 'top159_stacked_aux_integrated_bug_audit_latest.json'
OUT_DATA_TRUTH = REPORTS / 'top159_stacked_aux_integrated_data_truth_latest.json'
OUT_VERDICT_JSON = REPORTS / 'top159_stacked_aux_integrated_unique_verdict_latest.json'
OUT_VERDICT_MD = REPORTS / 'top159_stacked_aux_integrated_unique_verdict_latest.md'


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

base = load_module('crypto_pressure_for_stacked_aux', BASE_SCRIPT)
fill = load_module('fill_search_for_stacked_aux', FILL_SCRIPT)
aux = load_module('aux_research_for_stacked_aux', AUX_SCRIPT)
integrated = load_module('integrated_main_for_stacked_aux', INTEGRATED_SCRIPT)
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


def fit_classifier(engine: str, train: pd.DataFrame, features: list[str], params: dict[str, Any], label_col: str = 'label_up', random_labels: bool = False):
    x = train[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train[label_col].astype(int).to_numpy()
    if random_labels:
        rng = np.random.default_rng(RNG_SEED + len(train) + len(features))
        y = rng.permutation(y)
    if len(train) < int(params.get('min_train_rows', 500)) or len(np.unique(y)) < 2:
        return None
    if engine == 'lightgbm':
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=int(params.get('n_estimators', 120)), learning_rate=float(params.get('learning_rate', 0.035)),
            num_leaves=int(params.get('num_leaves', 24)), min_child_samples=int(params.get('min_child_samples', 60)),
            subsample=float(params.get('subsample', 0.88)), colsample_bytree=float(params.get('colsample_bytree', 0.88)),
            reg_lambda=float(params.get('reg_lambda', 1.0)), random_state=RNG_SEED, n_jobs=int(THREADS), verbose=-1,
        ).fit(x, y)
    if engine == 'xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=int(params.get('n_estimators', 120)), max_depth=int(params.get('depth', 4)), learning_rate=float(params.get('learning_rate', 0.035)),
            subsample=float(params.get('subsample', 0.88)), colsample_bytree=float(params.get('colsample_bytree', 0.88)),
            reg_lambda=float(params.get('reg_lambda', 1.0)), random_state=RNG_SEED, n_jobs=int(THREADS), eval_metric='logloss', verbosity=0,
        ).fit(x, y)
    if engine == 'catboost':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=int(params.get('n_estimators', 120)), depth=int(params.get('depth', 4)), learning_rate=float(params.get('learning_rate', 0.035)),
            l2_leaf_reg=float(params.get('reg_lambda', 1.0)), loss_function='Logloss', eval_metric='Logloss',
            random_seed=RNG_SEED, verbose=False, thread_count=int(THREADS),
        )
        return model.fit(x, y)
    if engine == 'logistic':
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=float(params.get('C', 1.0)), random_state=RNG_SEED))
        return model.fit(x, y)
    raise ValueError(engine)


def predict_prob(model: Any, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    x = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def oof_predict_frame(frame: pd.DataFrame, features: list[str], key: str, params: dict[str, Any], min_history_days: int = 365) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache_key = stable_hash({
        'key': key, 'params': params, 'blockDays': OOF_BLOCK_DAYS, 'trainDays': OOF_TRAIN_DAYS,
        'frameRows': len(frame), 'start': str(frame['dt'].min()), 'end': str(frame['dt'].max()), 'features': features[:80],
    })
    cache = CACHE_DIR / f'oof_{key}_{cache_key}.parquet'
    meta_path = CACHE_DIR / f'oof_{key}_{cache_key}.json'
    if cache.exists() and meta_path.exists():
        pred = pd.read_parquet(cache)
        meta = json.loads(meta_path.read_text())
        meta['cacheHit'] = True
        return pred, meta

    df = frame[['dt', 'label_up'] + features].copy().sort_values('dt').reset_index(drop=True)
    df['dt'] = pd.to_datetime(df['dt'], utc=True)
    start = df['dt'].min() + pd.Timedelta(days=min_history_days)
    end = df['dt'].max()
    blocks = []
    block_start = start.floor('D')
    fit_count = 0
    skipped = 0
    errors = []
    while block_start <= end:
        block_end = block_start + pd.Timedelta(days=OOF_BLOCK_DAYS)
        train_start = block_start - pd.Timedelta(days=OOF_TRAIN_DAYS)
        train = df[(df['dt'] < block_start) & (df['dt'] >= train_start)].copy()
        val = df[(df['dt'] >= block_start) & (df['dt'] < block_end)].copy()
        if len(val) == 0:
            block_start = block_end
            continue
        if len(train) > MAX_TRAIN_ROWS > 0:
            train = train.sort_values('dt').iloc[-MAX_TRAIN_ROWS:].copy()
        if len(train) < int(params.get('min_train_rows', 500)) or train['label_up'].nunique() < 2:
            skipped += 1
            block_start = block_end
            continue
        try:
            model = fit_classifier(params['engine'], train, features, params)
            if model is None:
                skipped += 1
                block_start = block_end
                continue
            prob = predict_prob(model, val, features)
            part = val[['dt', 'label_up']].copy()
            part[f'p_{key}_up'] = prob
            part[f'p_{key}_conf'] = np.maximum(prob, 1.0 - prob)
            part[f'pred_{key}_up'] = prob >= 0.5
            blocks.append(part)
            fit_count += 1
            if fit_count == 1 or fit_count % 5 == 0:
                print(f'[stacked] OOF {key}: fitted_blocks={fit_count} latest_block={block_start.date()} rows={sum(len(b) for b in blocks)}', flush=True)
        except Exception as exc:
            errors.append({'blockStart': str(block_start), 'error': str(exc)[:300]})
        block_start = block_end
    if blocks:
        out = pd.concat(blocks, ignore_index=True).sort_values('dt').reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=['dt', 'label_up', f'p_{key}_up', f'p_{key}_conf', f'pred_{key}_up'])
    meta = {
        'key': key, 'generatedAt': now_iso(), 'beijingTime': bj_now(), 'cacheHit': False,
        'engine': params['engine'], 'features': len(features), 'inputRows': len(df), 'predRows': len(out),
        'fitCount': fit_count, 'skippedBlocks': skipped, 'errors': errors[:10],
        'blockDays': OOF_BLOCK_DAYS, 'trainDays': OOF_TRAIN_DAYS,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    return out, meta


def aux_frame_for_key(key: str) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    if key in {'1h', '4h'}:
        raw = load_raw('ETH', key)
        df, feats = base.build_features(raw, key)
        clean = []
        for c in feats:
            if c in set(base.FORBIDDEN_FEATURES) or c.startswith(tuple(base.FORBIDDEN_PREFIXES)):
                continue
            if pd.to_numeric(df[c], errors='coerce').notna().sum() >= max(500, int(len(df) * 0.55)):
                clean.append(c)
        return df[['dt', 'label_up'] + clean].copy(), clean, {'source': f'raw_ETH_{key}', 'rows': len(df), 'featureCount': len(clean)}
    period = int(key[:-1])
    df, feats = aux.build_daily_period_features(period)
    clean = []
    for c in feats:
        if c in set(base.FORBIDDEN_FEATURES) or c.startswith(tuple(base.FORBIDDEN_PREFIXES)):
            continue
        if pd.to_numeric(df[c], errors='coerce').notna().sum() >= max(120, int(len(df) * 0.55)):
            clean.append(c)
    return df[['dt', 'label_up'] + clean].copy(), clean, {'source': f'rolling_{period}d_from_1h', 'rows': len(df), 'featureCount': len(clean)}


def build_aux_predictions() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    params = {
        'engine': os.environ.get('TOP159_STACKED_AUX_ENGINE', 'lightgbm'),
        'n_estimators': int(os.environ.get('TOP159_STACKED_AUX_ESTIMATORS', '120')),
        'learning_rate': float(os.environ.get('TOP159_STACKED_AUX_LR', '0.035')),
        'num_leaves': int(os.environ.get('TOP159_STACKED_AUX_LEAVES', '24')),
        'min_child_samples': int(os.environ.get('TOP159_STACKED_AUX_MIN_CHILD', '60')),
        'subsample': 0.88,
        'colsample_bytree': 0.88,
        'reg_lambda': 1.0,
        'depth': 4,
        'min_train_rows': 600,
    }
    preds = {}
    truth = {'generatedAt': now_iso(), 'beijingTime': bj_now(), 'auxParams': params, 'keys': {}, 'singleModelScoreboard': {}}
    for key in AUX_KEYS:
        print(f'[stacked] build OOF aux {key}...', flush=True)
        frame, feats, info = aux_frame_for_key(key)
        pred, meta = oof_predict_frame(frame, feats, key, params, min_history_days=365)
        preds[key] = pred
        truth['keys'][key] = {**info, **meta}
        # Score only where OOF predictions exist. This is not a live gate; it is a sanity audit.
        if len(pred):
            labels = pred['label_up'].astype(bool).to_numpy()
            prob = pd.to_numeric(pred[f'p_{key}_up'], errors='coerce').fillna(0.5).to_numpy()
            pred_up = prob >= 0.5
            acc = float((pred_up == labels).mean()) if len(labels) else 0.0
            truth['singleModelScoreboard'][key] = {
                'samples': int(len(pred)), 'accuracyPct': round(acc * 100, 6),
                'avgConfidence': round(float(np.maximum(prob, 1 - prob).mean()), 6),
            }
        else:
            truth['singleModelScoreboard'][key] = {'samples': 0, 'accuracyPct': 0.0, 'avgConfidence': 0.0}
    write_json(OUT_DATA_TRUTH, truth)
    return preds, truth


def merge_aux_asof(left: pd.DataFrame, pred: pd.DataFrame, key: str) -> pd.DataFrame:
    l = left.sort_values('dt').copy()
    r = pred[['dt', f'p_{key}_up', f'p_{key}_conf', f'pred_{key}_up']].copy().sort_values('dt')
    l['dt'] = pd.to_datetime(l['dt'], utc=True).astype('datetime64[ns, UTC]')
    r['dt'] = pd.to_datetime(r['dt'], utc=True).astype('datetime64[ns, UTC]')
    out = pd.merge_asof(l, r, on='dt', direction='backward')
    out[f'{key}_available'] = pd.to_numeric(out[f'p_{key}_up'], errors='coerce').notna().astype(int)
    out[f'p_{key}_up'] = pd.to_numeric(out[f'p_{key}_up'], errors='coerce').fillna(0.5)
    out[f'p_{key}_conf'] = pd.to_numeric(out[f'p_{key}_conf'], errors='coerce').fillna(0.5)
    out[f'pred_{key}_up'] = out[f'p_{key}_up'] >= 0.5
    return out


def add_oof_top159_score(df: pd.DataFrame, base_features: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    params = fill_params = aux.top159_params()
    feats = fill.feature_subset(base_features, params.get('feature_mode', 'trend'))
    oof_params = {
        'engine': params.get('engine', 'lightgbm'),
        'n_estimators': int(params.get('n_estimators', 160)), 'learning_rate': float(params.get('learning_rate', 0.035)),
        'num_leaves': int(params.get('num_leaves', 24)), 'min_child_samples': int(params.get('min_child_samples', 80)),
        'subsample': float(params.get('subsample', 0.88)), 'colsample_bytree': float(params.get('colsample_bytree', 0.88)),
        'reg_lambda': float(params.get('reg_lambda', 1.0)), 'depth': int(params.get('depth', 4)),
        'min_train_rows': 1000,
    }
    frame = df[['dt', 'label_up'] + feats].copy()
    pred, meta = oof_predict_frame(frame, feats, 'top159', oof_params, min_history_days=365)
    out = merge_aux_asof(df, pred, 'top159')
    p = pd.to_numeric(out['p_top159_up'], errors='coerce').fillna(0.5)
    out['top159_oof_score'] = np.maximum(p, 1 - p)
    out['top159_oof_pred_up'] = p >= 0.5
    meta['top159Params'] = params
    meta['featureCount'] = len(feats)
    return out, meta


def build_stacked_frame() -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    print('[stacked] build base integrated frame...', flush=True)
    raw15 = load_raw('ETH', '15m')
    df, feats15 = base.build_features(raw15, '15m')
    # Keep only clean 15m features.
    clean15 = []
    for c in feats15:
        if c in set(base.FORBIDDEN_FEATURES) or c.startswith(tuple(base.FORBIDDEN_PREFIXES)):
            continue
        s = pd.to_numeric(df[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
        if int(s.notna().sum()) >= max(500, int(len(df) * 0.55)):
            df[c] = s
            clean15.append(c)
    df = df.dropna(subset=['dt', 'label_up']).sort_values('dt').reset_index(drop=True)
    df, top_meta = add_oof_top159_score(df, clean15)
    preds, aux_truth = build_aux_predictions()
    for key in AUX_KEYS:
        df = merge_aux_asof(df, preds[key], key)
        p = pd.to_numeric(df[f'p_{key}_up'], errors='coerce').fillna(0.5)
        df[f'p_{key}_same_top159'] = np.where(df['top159_oof_pred_up'].astype(bool), p, 1 - p)
        df[f'p_{key}_oppose_top159'] = 1.0 - df[f'p_{key}_same_top159']
        df[f'{key}_disagree_top159'] = (df[f'p_{key}_same_top159'] < 0.5).astype(int)
    stack_feats = ['p_top159_up', 'top159_oof_score']
    for key in AUX_KEYS:
        stack_feats += [f'p_{key}_up', f'p_{key}_conf', f'p_{key}_same_top159', f'p_{key}_oppose_top159', f'{key}_available', f'{key}_disagree_top159']
    feature_cols = clean15 + stack_feats
    forbidden_hits = [c for c in feature_cols if c in set(base.FORBIDDEN_FEATURES) or c.startswith(tuple(base.FORBIDDEN_PREFIXES))]
    data_truth = {
        'generatedAt': now_iso(), 'beijingTime': bj_now(), 'liveMutation': 'none',
        'raw15Rows': int(len(raw15)), 'baseRows': int(len(df)), 'base15Features': len(clean15),
        'top159Oof': top_meta, 'auxTruth': aux_truth, 'featureCountBeforeCombo': len(feature_cols),
        'forbiddenFeatureHits': forbidden_hits,
        'timing': 'all p_* features are walk-forward/OOF and then merge_asof backward into 15m rows',
    }
    write_json(OUT_DATA_TRUTH, data_truth)
    return df, feature_cols, data_truth


def combo_features(combo: tuple[str, ...], mode: str) -> list[str]:
    feats = ['p_top159_up', 'top159_oof_score']
    for key in combo:
        feats += [f'p_{key}_up', f'p_{key}_conf', f'p_{key}_same_top159', f'p_{key}_oppose_top159', f'{key}_available', f'{key}_disagree_top159']
    feats += [f'combo_{mode}_mean_same', f'combo_{mode}_min_same', f'combo_{mode}_oppose_count', f'combo_{mode}_support']
    return feats


def add_combo_columns(df: pd.DataFrame, combo: tuple[str, ...], mode: str) -> pd.DataFrame:
    out = df.copy()
    same_cols = []
    for key in combo:
        same = pd.to_numeric(out[f'p_{key}_same_top159'], errors='coerce').fillna(0.5).to_numpy(dtype=float)
        if mode == 'reverse':
            same = 1 - same
        elif mode == 'short_same_long_reverse':
            rank = {'1h': 1, '4h': 2, '4d': 3, '7d': 4, '18d': 5}[key]
            if rank >= 4:
                same = 1 - same
        elif mode == 'short_reverse_long_same':
            rank = {'1h': 1, '4h': 2, '4d': 3, '7d': 4, '18d': 5}[key]
            if rank <= 3:
                same = 1 - same
        c = f'__same_{key}_{mode}'
        out[c] = same
        same_cols.append(c)
    mat = out[same_cols].to_numpy(dtype=float) if same_cols else np.zeros((len(out), 1)) + 0.5
    out[f'combo_{mode}_mean_same'] = mat.mean(axis=1)
    out[f'combo_{mode}_min_same'] = mat.min(axis=1)
    out[f'combo_{mode}_oppose_count'] = (mat < 0.45).sum(axis=1)
    out[f'combo_{mode}_support'] = (2 * mat - 1).mean(axis=1)
    return out.drop(columns=same_cols, errors='ignore')


def split_train_val(df: pd.DataFrame, window: str, train_window: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = df['dt'].max()
    days = 180 if window == '180d' else 365
    start = end - pd.Timedelta(days=days)
    val = df[df['dt'] >= start].copy()
    if train_window == 'full':
        train = df[df['dt'] < start].copy()
    else:
        train_days = {'1y': 365, '2y': 730, '3y': 1095, '5y': 1825, '7y': 2555}[train_window]
        train = df[(df['dt'] < start) & (df['dt'] >= start - pd.Timedelta(days=train_days))].copy()
    if MAX_TRAIN_ROWS > 0 and len(train) > MAX_TRAIN_ROWS:
        train = train.sort_values('dt').iloc[-MAX_TRAIN_ROWS:].copy()
    return train, val


def feature_subset(all_features: list[str], params: dict[str, Any]) -> list[str]:
    combo = tuple(params['combo'])
    mode = params['combo_mode']
    base_mode = params.get('base_mode', 'base15_core')
    if base_mode == 'prob_only':
        base_feats = []
    elif base_mode == 'base15_core':
        wanted = {'ret_1','ret_2','ret_4','ret_8','ret_16','ret_32','vol_8','vol_16','vol_32','range_8','range_16','ema_8_32','ema_16_64','ema_dist_8','ema_dist_16','ema_dist_32','rsi_14','bb_pos','hour_sin','hour_cos','dow_sin','dow_cos'}
        base_feats = [c for c in all_features if c in wanted]
    else:
        base_feats = [c for c in all_features if not c.startswith(('p_', 'top159_', 'combo_', '1h_', '4h_', '4d_', '7d_', '18d_'))]
    feats = base_feats + combo_features(combo, mode)
    seen = []
    for c in feats:
        if c in all_features and c not in seen:
            seen.append(c)
    return seen[:MAX_FEATURES]


def fit_main(engine: str, train: pd.DataFrame, feats: list[str], params: dict[str, Any], random_labels: bool = False):
    p = dict(params)
    p.setdefault('min_train_rows', 1000)
    return fit_classifier(engine, train, feats, p, random_labels=random_labels)


def select_rows(val: pd.DataFrame, prob: np.ndarray, params: dict[str, Any]) -> pd.DataFrame:
    pred_up = prob >= 0.5
    score = np.maximum(prob, 1.0 - prob)
    mask = score >= 0.5 + float(params['edge'])
    if params.get('min_top159_score', 0.0) > 0:
        mask &= pd.to_numeric(val['top159_oof_score'], errors='coerce').fillna(0.5).to_numpy() >= float(params['min_top159_score'])
    if params.get('combo_min_same', 0.0) > 0:
        mask &= pd.to_numeric(val[f"combo_{params['combo_mode']}_mean_same"], errors='coerce').fillna(0.5).to_numpy() >= float(params['combo_min_same'])
    out = val.loc[mask, ['dt', 'label_up']].copy().reset_index(drop=True)
    if len(out) == 0:
        return out.assign(pred_up15=[], score15=[], won=[])
    pred_sel = pred_up[mask]
    score_sel = score[mask]
    out['pred_up15'] = pred_sel.astype(bool)
    out['score15'] = score_sel
    out['won'] = out['pred_up15'].to_numpy() == out['label_up'].astype(bool).to_numpy()
    return out


def evaluate_params(df: pd.DataFrame, all_features: list[str], params: dict[str, Any], name: str) -> dict[str, Any] | None:
    rows = []
    combo = tuple(params['combo'])
    mode = params['combo_mode']
    dfx = add_combo_columns(df, combo, mode)
    feats = feature_subset(list(dict.fromkeys(all_features + combo_features(combo, mode))), params)
    if len(feats) < 8:
        return None
    for window in ['180d', '365d']:
        train, val = split_train_val(dfx, window, params['train_window'])
        model = fit_main(params['engine'], train, feats, params)
        if model is None:
            return None
        prob = predict_prob(model, val, feats)
        selected = select_rows(val, prob, params)
        if len(selected) == 0:
            return None
        row = aux.curve_metrics(selected, name, window, 'stacked_aux_integrated')
        row.update({
            'engine': params['engine'], 'trainWindow': params['train_window'], 'edge': params['edge'],
            'combo': '+'.join(combo), 'comboMode': mode, 'featureCount': len(feats),
            'baseMode': params.get('base_mode'), 'minTop159Score': params.get('min_top159_score', 0.0),
            'comboMinSame': params.get('combo_min_same', 0.0), 'baseTrades': None, 'retentionPct': None,
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
    # FAK pressure is a hard sanity gate, but its compounded bankroll can become
    # enormous and swamp the risk-first objective. Keep it logarithmic so the
    # selector prefers robust toxicity curves rather than a single explosive
    # proxy number.
    fak_log = sum(math.log1p(max(0.0, by[w]['fak2EndingBankroll']) / 850.0) for w in ['180d', '365d'])
    dd = by['365d']['maxDrawdownP50']
    mp = by['180d']['monthlyPositiveRatioP50'] + by['365d']['monthlyPositiveRatioP50']
    return gain_p50 * 2.0 + gain_p5 * 1.5 + fak_log * 250.0 + mp * 300.0 - dd * 0.8


def all_combos() -> list[tuple[str, ...]]:
    out = []
    for r in range(1, len(AUX_KEYS) + 1):
        out.extend(tuple(x) for x in itertools.combinations(AUX_KEYS, r))
    return out


def param_grid() -> list[dict[str, Any]]:
    engines = [e for e in ENGINES if e in {'lightgbm', 'xgboost', 'catboost', 'logistic'}]
    rows = []
    combos = all_combos()
    combo_modes = ['same', 'reverse', 'short_same_long_reverse', 'short_reverse_long_same']
    for combo, mode, engine in itertools.product(combos, combo_modes, engines):
        for train_window in ['1y', '3y', '5y', 'full']:
            for base_mode in ['prob_only', 'base15_core']:
                for edge in [0.03, 0.045, 0.06, 0.075]:
                    for mts in [0.0, 0.545]:
                        for cms in [0.0, 0.47]:
                            if engine == 'logistic':
                                rows.append({'engine': engine, 'train_window': train_window, 'combo': combo, 'combo_mode': mode, 'base_mode': base_mode, 'edge': edge, 'min_top159_score': mts, 'combo_min_same': cms, 'n_estimators': 1, 'learning_rate': 0.04, 'num_leaves': 16, 'min_child_samples': 80, 'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_lambda': 1.0, 'depth': 3, 'C': 1.0})
                                continue
                            for ne, lr, leaves, mcs, reg in [(120, 0.035, 24, 80, 0.8)]:
                                rows.append({'engine': engine, 'train_window': train_window, 'combo': combo, 'combo_mode': mode, 'base_mode': base_mode, 'edge': edge, 'min_top159_score': mts, 'combo_min_same': cms, 'n_estimators': ne, 'learning_rate': lr, 'num_leaves': leaves, 'min_child_samples': mcs, 'subsample': 0.88, 'colsample_bytree': 0.88, 'reg_lambda': reg, 'depth': 4})
    # Stage ordering: explicit combos first, then all remaining. This gives early signal without excluding anything.
    priority = [('4d',), ('1h',), ('4h',), ('7d',), ('18d',), ('4d','1h'), ('4d','7d','1h'), ('1h','7d','18d'), ('4d','18d','4h'), ('1h','4h','4d','7d','18d')]
    def rank(p: dict[str, Any]) -> tuple[int, int, str]:
        combo = tuple(p['combo'])
        try:
            pr = priority.index(combo)
        except ValueError:
            pr = 999
        return pr, len(combo), '+'.join(combo)
    rows.sort(key=rank)
    if PARAM_LIMIT > 0:
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
    name = f"stacked_{params['engine']}_{params['train_window']}_{'+'.join(params['combo'])}_{params['combo_mode']}_{params['base_mode']}_edge{params['edge']}_{stable_hash(params)}"
    try:
        out = evaluate_params(_DF, _FEATURES, params, name)
        if out is None:
            return {'name': name, 'params': params, 'error': 'empty_or_fit_failed'}
        return out
    except Exception as exc:
        return {'name': name, 'params': params, 'error': str(exc)[:500]}


def bug_audit(df: pd.DataFrame, features: list[str], base_by_window: dict[str, dict[str, Any]], data_truth: dict[str, Any]) -> dict[str, Any]:
    forbidden = [c for c in features if c in set(base.FORBIDDEN_FEATURES) or c.startswith(tuple(base.FORBIDDEN_PREFIXES))]
    repeat = []
    sample_params = {'engine': 'logistic', 'train_window': '5y', 'combo': ('1h','4d'), 'combo_mode': 'same', 'base_mode': 'prob_only', 'edge': 0.045, 'min_top159_score': 0.0, 'combo_min_same': 0.0, 'n_estimators': 1, 'learning_rate': 0.04, 'num_leaves': 16, 'min_child_samples': 80, 'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_lambda': 1.0, 'depth': 3, 'C': 1.0}
    c1 = evaluate_params(df, features, sample_params, 'stacked_audit_repeat')
    c2 = evaluate_params(df, features, sample_params, 'stacked_audit_repeat')
    if c1 and c2:
        for a, b in zip(c1['rows'], c2['rows']):
            repeat.append({'window': a['window'], 'hash1': a['setHash'], 'hash2': b['setHash'], 'p50_1': a['endingBankrollP50'], 'p50_2': b['endingBankrollP50'], 'passed': a['setHash'] == b['setHash'] and a['endingBankrollP50'] == b['endingBankrollP50']})
    random_label = {'status': 'skipped'}
    try:
        dfx = add_combo_columns(df, ('1h','4d'), 'same')
        feats = feature_subset(list(dict.fromkeys(features + combo_features(('1h','4d'), 'same'))), sample_params)
        train, val = split_train_val(dfx, '180d', '5y')
        model = fit_main('lightgbm', train, feats, {**sample_params, 'engine':'lightgbm', 'n_estimators':80, 'min_train_rows':1000}, random_labels=True)
        if model is not None:
            prob = predict_prob(model, val, feats)
            selected = select_rows(val, prob, {**sample_params, 'edge': 0.045})
            wr = float(selected['won'].mean()) if len(selected) else 0.0
            random_label = {'status': 'ok', 'selectedTrades': int(len(selected)), 'winRatePct': round(wr * 100, 6), 'passed': wr < 0.57 or len(selected) < 100}
    except Exception as exc:
        random_label = {'status': 'error', 'error': str(exc)[:300], 'passed': False}
    audit = {
        'generatedAt': now_iso(), 'beijingTime': bj_now(), 'liveMutation': 'none',
        'forbiddenFeatureHits': forbidden,
        'repeatability': repeat,
        'repeatabilityPassed': all(x['passed'] for x in repeat) if repeat else False,
        'randomLabelAudit': random_label,
        'windowIsolation': '180d/365d split and curve metrics are computed independently per candidate',
        'oofTiming': data_truth.get('timing'),
        'baseRows': base_by_window,
    }
    audit['passed'] = not forbidden and audit['repeatabilityPassed'] and bool(random_label.get('passed', False))
    write_json(OUT_AUDIT_JSON, audit)
    return audit


def write_progress(results: list[dict[str, Any]], base_by_window: dict[str, dict[str, Any]], total: int, started: float, data_truth: dict[str, Any], finished: bool = False):
    valid = [r for r in results if r.get('rows')]
    verdicts = []
    for c in valid:
        ok, reasons = pass_gate(c, base_by_window)
        verdicts.append({**c, 'passed': ok, 'reasons': reasons, 'score': score_candidate(c, base_by_window)})
    verdicts.sort(key=lambda x: x['score'], reverse=True)
    strict = [v for v in verdicts if v['passed']]
    payload = {
        'generatedAt': now_iso(), 'beijingTime': bj_now(), 'finished': finished,
        'elapsedSeconds': round(time.time() - started, 3), 'workers': WORKERS, 'workerThreads': THREADS,
        'totalCandidates': total, 'doneCount': len(results), 'validCount': len(valid), 'strictPassCount': len(strict),
        'dataTruth': data_truth, 'baseRows': base_by_window, 'topCandidates': verdicts[:300], 'strictPass': strict[:100],
    }
    write_json(OUT_CHECKPOINT, payload)
    write_json(OUT_LEADERBOARD_JSON, {'generatedAt': payload['generatedAt'], 'baseRows': base_by_window, 'rows': verdicts[:500]})
    selected = strict[0] if strict else None
    verdict = {'generatedAt': payload['generatedAt'], 'status': 'candidate_passed_research_gate' if selected else 'running_or_no_stacked_candidate_yet', 'selected': selected, 'baseRows': base_by_window, 'liveAction': 'research_only_no_live_change'}
    write_json(OUT_VERDICT_JSON, verdict)
    lines = ['# top159 五模型概率一体化搜索', '', f'- 北京时间：`{payload["beijingTime"]}`', f'- 完成：`{len(results)}/{total}`', f'- 严格通过：`{len(strict)}`', '- live动作：`research_only_no_live_change`', '', '|候选|窗口|组合|模型|训练窗|交易数|胜/负|胜率|P5|P50|P95|最大回撤|月正收益|FAK+2资金|特征|edge|哈希|', '|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    def add_row(r):
        lines.append(f"|{r['name']}|{r['window']}|{r.get('combo','-')}|{r.get('engine','baseline')}|{r.get('trainWindow','-')}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']}|{r['endingBankrollP5']}|{r['endingBankrollP50']}|{r['endingBankrollP95']}|{r['maxDrawdownP50']}|{r['monthlyPositiveRatioP50']}|{r['fak2EndingBankroll']}|{r.get('featureCount','-')}|{r.get('edge','-')}|`{r['setHash']}`|")
    for r in [base_by_window['180d'], base_by_window['365d']]:
        add_row(r)
    for c in verdicts[:25]:
        for r in c['rows']:
            add_row(r)
    write_text(OUT_LEADERBOARD_MD, '\n'.join(lines) + '\n')
    vlines = ['# top159 五模型概率一体化唯一结论', '', f'- 北京时间：`{payload["beijingTime"]}`', f'- 状态：`{verdict["status"]}`', '- live动作：`research_only_no_live_change`']
    if selected:
        vlines += [f'- 选中候选：`{selected["name"]}`', f'- 参数：`{json.dumps(selected["params"], ensure_ascii=False)}`']
    else:
        vlines += ['- 暂无过门候选；搜索继续或等待下一轮特征/模型扩展。']
    write_text(OUT_VERDICT_MD, '\n'.join(vlines) + '\n')


def run() -> int:
    started = time.time()
    print(f'[stacked] start {bj_now()} workers={WORKERS} maxSeconds={MAX_SECONDS}', flush=True)
    df, features, data_truth = build_stacked_frame()
    base_by_window = base_rows()
    audit = bug_audit(df, features, base_by_window, data_truth)
    data_truth = {**data_truth, 'bugAudit': audit, 'comboCount': len(all_combos()), 'auxKeys': AUX_KEYS}
    print(f'[stacked] rows={len(df)} features={len(features)} auditPassed={audit["passed"]}', flush=True)
    params = param_grid()
    total = len(params)
    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    results = []
    done_names = set()
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
        name = f"stacked_{p['engine']}_{p['train_window']}_{'+'.join(p['combo'])}_{p['combo_mode']}_{p['base_mode']}_edge{p['edge']}_{stable_hash(p)}"
        if name not in done_names:
            pending.append((i, p))
    print(f'[stacked] total={total} done={len(results)} pending={len(pending)}', flush=True)
    write_progress(results, base_by_window, total, started, data_truth, finished=False)
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
                    write_progress(results, base_by_window, total, started, data_truth, finished=False)
                    print(f'[stacked] checkpoint {len(results)}/{total}', flush=True)
                    last = time.time()
            for fut in futures:
                fut.cancel()
    finished = len([r for r in results if r.get('rows') or r.get('error')]) >= total
    write_progress(results, base_by_window, total, started, data_truth, finished=finished)
    print(json.dumps({'status': 'finished' if finished else 'checkpointed', 'done': len(results), 'total': total, 'leaderboard': str(OUT_LEADERBOARD_MD), 'verdict': str(OUT_VERDICT_MD)}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(run())
