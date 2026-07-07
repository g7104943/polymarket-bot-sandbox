#!/usr/bin/env python3
from __future__ import annotations

"""top159 execution-quality / orderbook-quality research.

Research only. Does not mutate live configs and does not submit orders.

Main purpose:
  1) downgrade old P50/FAK+2 style numbers to proxy status,
  2) separate strong real fills, medium orderbook evidence, weak token-path proxy,
  3) test whether simple execution-quality gates/models improve top159 without
     pretending ideal fills are live-tradable.
"""

import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path('/Users/mac/polyfun')
NEXT = ROOT / 'polyfun-next'
SCRIPTS = NEXT / 'scripts'
REPORTS = ROOT / 'reports'
DATA = ROOT / 'data' / 'processed'
RUNTIME = NEXT / 'runtime'

EXTREME_SCRIPT = SCRIPTS / 'run_top159_integrated_main_extreme_search.py'
ARCHIVE_SCRIPT = ROOT / 'scripts' / 'ops' / 'run_top159_all_archived_real_eth_compare.py'
EPISODES = DATA / 'vnext_entry_exit_episodes_eth_usdt.parquet'
LOCAL_ORDERBOOK = DATA / 'vnext_execution_orderbook_eth_usdt.jsonl'
LIFECYCLE = DATA / 'vnext_execution_lifecycle_eth_usdt.jsonl'
OFFICIAL_DRYRUN = RUNTIME / 'top159_official_orderbook_dryrun.jsonl'
CANARY_LEDGER = RUNTIME / 'canary_ledger.jsonl'
LIVE_PROFILE = RUNTIME / 'top159_live_model_profile.json'

START_BANKROLL = 850.0
STAKE_FRAC = 0.01
RNG_SEED = 20260504

OUT_DATA_JSON = REPORTS / 'top159_execution_quality_data_truth_latest.json'
OUT_DATA_MD = REPORTS / 'top159_execution_quality_data_truth_latest.md'
OUT_COMPARE_JSON = REPORTS / 'top159_execution_quality_model_compare_latest.json'
OUT_COMPARE_MD = REPORTS / 'top159_execution_quality_model_compare_latest.md'
OUT_VERDICT_JSON = REPORTS / 'top159_execution_quality_unique_verdict_latest.json'
OUT_VERDICT_MD = REPORTS / 'top159_execution_quality_unique_verdict_latest.md'


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

extreme = load_module('execution_quality_extreme', EXTREME_SCRIPT)
archive = load_module('execution_quality_archive', ARCHIVE_SCRIPT)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def bj_now() -> str:
    return pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y-%m-%d %H:%M:%S CST')


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def parse_market_start(slug: str) -> pd.Timestamp | None:
    try:
        return pd.to_datetime(int(str(slug).rsplit('-', 1)[-1]), unit='s', utc=True)
    except Exception:
        return None


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def loads_json_list(x: Any) -> list[Any]:
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, str):
        try:
            out = json.loads(x)
            return out if isinstance(out, list) else []
        except Exception:
            return []
    return []


def max_drawdown_from_equity(equity: Iterable[float]) -> float:
    arr = np.asarray(list(equity), dtype=float)
    if arr.size == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    return float(np.max(peak - arr))


def compound_curve(won: np.ndarray, buy_price: np.ndarray, fill_frac: np.ndarray | None = None) -> tuple[float, float, list[float], list[float]]:
    eq = START_BANKROLL
    curve = [eq]
    pnl_values: list[float] = []
    if fill_frac is None:
        fill_frac = np.ones_like(buy_price, dtype=float)
    for ok, px, ff in zip(won.astype(bool), buy_price.astype(float), fill_frac.astype(float)):
        if not np.isfinite(px) or px <= 0 or ff <= 0:
            curve.append(eq)
            pnl_values.append(0.0)
            continue
        stake = eq * STAKE_FRAC * min(1.0, max(0.0, float(ff)))
        ret = (1.0 / float(px) - 1.0) if ok else -1.0
        pnl = stake * ret
        eq = max(0.0, eq + pnl)
        curve.append(eq)
        pnl_values.append(float(pnl))
    return float(eq), max_drawdown_from_equity(curve), curve, pnl_values


def label_from_raw_15m(dt: pd.Timestamp) -> bool | None:
    try:
        raw = extreme.load_raw('ETH', '15m')
        if 'date' in raw.columns:
            raw_dt = pd.to_datetime(raw['date'], utc=True, errors='coerce')
        else:
            ts = pd.to_numeric(raw['timestamp'], errors='coerce')
            unit = 'ms' if float(ts.dropna().median()) > 10_000_000_000 else 's'
            raw_dt = pd.to_datetime(ts, unit=unit, utc=True, errors='coerce')
        tmp = raw.copy()
        tmp['dt'] = raw_dt
        row = tmp[tmp['dt'] == pd.Timestamp(dt)]
        if row.empty:
            return None
        r = row.iloc[0]
        return bool(float(r['close']) > float(r['open']))
    except Exception:
        return None


_RAW_LABEL_CACHE: dict[pd.Timestamp, bool | None] = {}

def actual_up_for_dt(dt: pd.Timestamp) -> bool | None:
    t = pd.Timestamp(dt).tz_convert('UTC') if pd.Timestamp(dt).tzinfo else pd.Timestamp(dt, tz='UTC')
    if t not in _RAW_LABEL_CACHE:
        _RAW_LABEL_CACHE[t] = label_from_raw_15m(t)
    return _RAW_LABEL_CACHE[t]


@dataclass
class MetricRow:
    window: str
    method: str
    truth_tier: str
    trades: int
    executable: int
    wins: int
    losses: int
    pending: int
    win_rate_pct: float
    pnl: float
    ending_funds: float
    max_drawdown: float
    avg_buy_price: float
    avg_spread_cost: float
    winner_fill_rate_pct: float
    loser_fill_rate_pct: float
    winner_avg_fill_fraction: float
    loser_avg_fill_fraction: float
    official_missing: int
    orderbook_missing: int
    retention_pct: float
    set_hash: str
    note: str


def metric_from_proxy(df: pd.DataFrame, method: str, window: str, mask: pd.Series, price_col: str, fill_col: str | None, note: str) -> MetricRow:
    base_n = int(len(df))
    sub = df[mask].copy()
    if sub.empty:
        return MetricRow(window, method, 'weak_token_path_proxy', base_n, 0, 0, 0, 0, 0.0, 0.0, START_BANKROLL, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, base_n, 0.0, 'empty', note)
    price = pd.to_numeric(sub[price_col], errors='coerce').to_numpy(dtype=float)
    fill = pd.to_numeric(sub[fill_col], errors='coerce').fillna(1.0).clip(0, 1).to_numpy(dtype=float) if fill_col else np.ones(len(sub), dtype=float)
    valid = np.isfinite(price) & (price > 0) & (fill > 0)
    exe = sub[valid].copy()
    price = price[valid]
    fill = fill[valid]
    won = exe['won'].astype(bool).to_numpy() if len(exe) else np.array([], dtype=bool)
    ending, dd, _, pnl_values = compound_curve(won, price, fill)
    wins = int(won.sum())
    losses = int(len(won) - wins)
    winners = sub[sub['won'].astype(bool)]
    losers = sub[~sub['won'].astype(bool)]
    win_filled = exe[exe['won'].astype(bool)] if len(exe) else exe
    loss_filled = exe[~exe['won'].astype(bool)] if len(exe) else exe
    spread_vals = pd.to_numeric(sub.get('ob_spread_ratio_last', pd.Series(dtype=float)), errors='coerce')
    return MetricRow(
        window=window,
        method=method,
        truth_tier='weak_token_path_proxy',
        trades=base_n,
        executable=int(len(exe)),
        wins=wins,
        losses=losses,
        pending=0,
        win_rate_pct=round(100.0 * wins / len(won), 6) if len(won) else 0.0,
        pnl=round(float(sum(pnl_values)), 6),
        ending_funds=round(ending, 6),
        max_drawdown=round(dd, 6),
        avg_buy_price=round(float(np.nanmean(price)), 6) if len(price) else 0.0,
        avg_spread_cost=round(float(spread_vals.mean()), 6) if len(spread_vals) else 0.0,
        winner_fill_rate_pct=round(100.0 * len(win_filled) / len(winners), 6) if len(winners) else 0.0,
        loser_fill_rate_pct=round(100.0 * len(loss_filled) / len(losers), 6) if len(losers) else 0.0,
        winner_avg_fill_fraction=round(float(pd.to_numeric(win_filled.get('proxy_fill_fraction', 1.0), errors='coerce').fillna(1.0).mean()), 6) if len(win_filled) else 0.0,
        loser_avg_fill_fraction=round(float(pd.to_numeric(loss_filled.get('proxy_fill_fraction', 1.0), errors='coerce').fillna(1.0).mean()), 6) if len(loss_filled) else 0.0,
        official_missing=0,
        orderbook_missing=int(base_n - len(exe)),
        retention_pct=round(100.0 * len(sub) / base_n, 6) if base_n else 0.0,
        set_hash=stable_hash(exe[['dt','pred_up15','won']].to_dict('records')) if len(exe) else 'empty',
        note=note,
    )


def build_current_live_candidates() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not LIVE_PROFILE.exists():
        raise RuntimeError(f'missing {LIVE_PROFILE}')
    profile = json.loads(LIVE_PROFILE.read_text())
    params = dict(profile['params'])
    df, features, truth = extreme.build_integrated_frame()
    feats = extreme.feature_subset(features, params['feature_mode'])
    if params.get('max_features'):
        feats = feats[:int(params['max_features'])]
    out_rows = []
    for window in ['180d', '365d']:
        train, val = extreme.split_train_val(df, window, params['train_window'])
        model = extreme.fit_model(params['engine'], train, feats, params)
        if model is None:
            raise RuntimeError(f'fit failed for {window}')
        prob = extreme.predict(model, val, feats)
        selected = extreme.select_rows(val, prob, params)
        selected['window_source'] = window
        out_rows.append(selected)
    cand = pd.concat(out_rows, ignore_index=True).drop_duplicates(['dt','pred_up15'], keep='last')
    cand['dt'] = pd.to_datetime(cand['dt'], utc=True)
    cand['market_slug'] = cand['dt'].map(lambda x: f"eth-updown-15m-{int(pd.Timestamp(x).timestamp())}")
    cand['model_name'] = profile.get('profile') or profile.get('name') or 'live_profile'
    cand['selected_candidate'] = profile.get('selectedCandidate') or profile.get('selected_candidate')
    cand['score15'] = pd.to_numeric(cand['score15'], errors='coerce')
    return cand.sort_values('dt').reset_index(drop=True), {'profile': profile, 'dataTruth': truth, 'featureCount': len(feats)}


def load_episode_proxy() -> pd.DataFrame:
    ep = pd.read_parquet(EPISODES)
    ep = ep.copy()
    ep['dt'] = pd.to_datetime(ep['timestamp'], utc=True, errors='coerce')
    keep_cols = ['dt','actual_up','path_t_sec_json','path_up_json','entry_price','entry_fill_status','entry_fill_fraction']
    keep_cols += [c for c in ep.columns if c.startswith('ob_')]
    keep_cols = [c for c in keep_cols if c in ep.columns]
    return ep[keep_cols].dropna(subset=['dt']).drop_duplicates('dt', keep='last').sort_values('dt')


def side_visible_price(row: pd.Series) -> float:
    vals = loads_json_list(row.get('path_up_json'))
    if not vals:
        return np.nan
    up = max(0.001, min(0.999, safe_float(vals[0])))
    return up if bool(row.get('pred_up15')) else (1.0 - up)


def join_proxy_features(candidates: pd.DataFrame) -> pd.DataFrame:
    ep = load_episode_proxy()
    out = candidates.merge(ep, on='dt', how='left')
    out['proxy_visible_price'] = out.apply(side_visible_price, axis=1)
    if 'entry_price' in out.columns:
        out['proxy_entry_price'] = pd.to_numeric(out['entry_price'], errors='coerce')
    else:
        out['proxy_entry_price'] = out['proxy_visible_price']
    if 'entry_fill_fraction' in out.columns:
        out['proxy_fill_fraction'] = pd.to_numeric(out['entry_fill_fraction'], errors='coerce').fillna(0.0).clip(0,1)
    else:
        out['proxy_fill_fraction'] = np.where(out['proxy_visible_price'].notna(), 1.0, 0.0)
    if 'entry_fill_status' in out.columns:
        out['proxy_filled'] = out['entry_fill_status'].astype(str).str.contains('filled', case=False, na=False)
    else:
        out['proxy_filled'] = out['proxy_fill_fraction'] > 0
    # proxy spread is ratio; create fallback visible spread bucket.
    for c in ['ob_spread_ratio_last','ob_spread_ratio_mean','ob_depth_imbalance_last','ob_bid_ask_imbalance_last','ob_pressure_last']:
        if c not in out.columns:
            out[c] = np.nan
    return out


def parse_official_dryrun() -> pd.DataFrame:
    rows = []
    if not OFFICIAL_DRYRUN.exists():
        return pd.DataFrame()
    for line in OFFICIAL_DRYRUN.read_text(errors='ignore').splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        sig = d.get('signal') or {}
        m = d.get('market') or {}
        ob = d.get('orderbook') or {}
        slug = m.get('market_slug') or sig.get('market_slug')
        if not slug:
            continue
        start = parse_market_start(slug)
        if start is None:
            continue
        actual_up = actual_up_for_dt(start)
        side = str(sig.get('side') or '').upper()
        pred_up = side == 'UP'
        won = (actual_up is not None and pred_up == bool(actual_up))
        rows.append({
            'observed_at': pd.to_datetime(d.get('observedAt'), utc=True, errors='coerce'),
            'market_slug': slug,
            'dt': start,
            'side': side,
            'pred_up15': pred_up,
            'model_score': safe_float(sig.get('modelScore')),
            'won': bool(won) if actual_up is not None else np.nan,
            'best_ask': safe_float(ob.get('bestAsk')),
            'best_bid': safe_float(ob.get('bestBid')),
            'spread': safe_float(ob.get('spread')),
            'ask_depth': safe_float(ob.get('askDepthTop3Shares')),
            'bid_depth': safe_float(ob.get('bidDepthTop3Shares')),
            'elapsed_sec': safe_float((d.get('entryWindow') or {}).get('elapsedSeconds')),
            'entry_allowed': bool((d.get('entryWindow') or {}).get('allowed')),
            'source': 'official_current_orderbook_dryrun',
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).dropna(subset=['observed_at','dt']).sort_values('observed_at')


def parse_live_official_fills() -> pd.DataFrame:
    rows = []
    if not CANARY_LEDGER.exists():
        return pd.DataFrame()
    for line in CANARY_LEDGER.read_text(errors='ignore').splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get('event_type') != 'order_official_confirmation':
            continue
        payload = d.get('payload') or {}
        status = payload.get('status') or {}
        if status.get('truth') != 'official_filled':
            continue
        cand = payload.get('candidate') or (payload.get('plan') or {}).get('candidate') or {}
        raw = status.get('raw') or {}
        slug = cand.get('market_slug')
        start = parse_market_start(slug or '')
        if start is None:
            continue
        actual_up = actual_up_for_dt(start)
        side = str(cand.get('side') or raw.get('outcome') or '').upper()
        pred_up = side in {'UP', 'YES'}
        price = safe_float(raw.get('price'), safe_float((payload.get('plan') or {}).get('price')))
        shares = safe_float(status.get('matched_shares'), safe_float(raw.get('size_matched')))
        notional = price * shares if np.isfinite(price) and np.isfinite(shares) else safe_float((payload.get('plan') or {}).get('notional_usd'))
        won = (actual_up is not None and pred_up == bool(actual_up))
        pnl = notional * ((1.0 / price - 1.0) if won and price > 0 else -1.0) if np.isfinite(notional) and np.isfinite(price) and price > 0 and actual_up is not None else np.nan
        rows.append({
            'ts': pd.to_datetime(d.get('ts'), utc=True, errors='coerce'),
            'market_slug': slug,
            'dt': start,
            'side': side,
            'pred_up15': pred_up,
            'model_score': safe_float(cand.get('model_score')),
            'price': price,
            'shares': shares,
            'notional': notional,
            'won': bool(won) if actual_up is not None else np.nan,
            'pnl': pnl,
            'strategy_profile': cand.get('strategy_profile'),
            'shock_candidate_id': cand.get('shock_candidate_id'),
            'order_id': status.get('order_id'),
            'source': 'official_filled_live_top159',
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).dropna(subset=['ts','dt']).sort_values('ts')


def summarize_live_fills(df: pd.DataFrame) -> MetricRow:
    if df.empty:
        return MetricRow('live_all', 'live_official_fills', 'strong_official_fills', 0, 0, 0, 0, 0, 0, 0, START_BANKROLL, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'empty', 'top159 official matched orders only')
    d = df.dropna(subset=['price']).copy()
    resolved = d[d['won'].notna()].copy()
    wins = int(resolved['won'].astype(bool).sum()) if len(resolved) else 0
    losses = int(len(resolved) - wins)
    pending = int(len(d) - len(resolved))
    pnl = float(pd.to_numeric(resolved['pnl'], errors='coerce').fillna(0).sum()) if len(resolved) else 0.0
    curve = START_BANKROLL + pd.to_numeric(resolved['pnl'], errors='coerce').fillna(0).cumsum()
    dd = max_drawdown_from_equity([START_BANKROLL] + curve.tolist())
    return MetricRow(
        window='live_all', method='top159真钱官方成交', truth_tier='strong_official_fills', trades=int(len(d)), executable=int(len(d)),
        wins=wins, losses=losses, pending=pending, win_rate_pct=round(100*wins/len(resolved),6) if len(resolved) else 0,
        pnl=round(pnl,6), ending_funds=round(START_BANKROLL+pnl,6), max_drawdown=round(dd,6),
        avg_buy_price=round(float(d['price'].mean()),6), avg_spread_cost=0.0,
        winner_fill_rate_pct=100.0, loser_fill_rate_pct=100.0,
        winner_avg_fill_fraction=1.0, loser_avg_fill_fraction=1.0,
        official_missing=0, orderbook_missing=pending, retention_pct=100.0,
        set_hash=stable_hash(d[['market_slug','side','won','price']].to_dict('records')),
        note=f'强真相：官网确认 MATCHED={len(d)}；已结算/可标胜负={len(resolved)}；未结算或缺标签={pending}。未成交候选不在本行。'
    )


def archived_rows_for_candidate(candidates: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        fill_rows, scan_audit = archive.load_live_fill_rows()
        all_eth = archive.aggregate_logical(fill_rows, '全部归档ETH15m live去重')
        strict_mask = fill_rows['sourcePath'].str.contains('slot01|forcedtrade|base_forcedtrade|__live_eth|logs_v5_exp10_bp0480__live_eth', case=False, regex=True)
        strict = archive.aggregate_logical(fill_rows[strict_mask].copy(), '严格旧slot1/ETH相关live去重') if strict_mask.any() else all_eth.iloc[0:0]
        rows=[]
        pred = candidates[['dt','pred_up15','score15']].drop_duplicates('dt', keep='last')
        for scope, old in [('严格旧slot1/ETH相关live去重', strict), ('全部归档ETH15m live去重', all_eth)]:
            if old.empty:
                continue
            merged = old.merge(pred, left_on='marketStart', right_on='dt', how='left')
            chosen = merged[merged['pred_up15'].notna()].copy().sort_values('marketStart')
            if chosen.empty:
                rows.append({'scope': scope, 'oldRealMarkets': int(len(old)), 'selectedTrades': 0})
                continue
            won = chosen['pred_up15'].astype(bool).to_numpy() == chosen['actualUp'].astype(bool).to_numpy()
            one = np.where(won, 1.0, -1.0)
            fak52 = np.where(won, 1.0/0.52 - 1.0, -1.0)
            same_dir = chosen['pred_up15'].astype(bool).to_numpy() == (chosen['direction'].astype(str).str.upper() == 'UP').to_numpy()
            same = chosen[same_dir]
            rows.append({
                'scope': scope,
                'oldRealMarkets': int(len(old)),
                'selectedTrades': int(len(chosen)),
                'wins': int(won.sum()),
                'losses': int(len(won)-int(won.sum())),
                'winRatePct': round(100*float(won.mean()),6),
                'oneUnitPnl': round(float(one.sum()),6),
                'oneUnitMaxDrawdown': round(max_drawdown_from_equity(np.concatenate([[0], np.cumsum(one)])),6),
                'fak52Pnl': round(float(fak52.sum()),6),
                'sameDirectionExecutableTrades': int(len(same)),
                'sameDirectionActualPnlUsd': round(float(same['pnl'].astype(float).sum()),6) if len(same) else 0.0,
                'truthLimit': '不是新模型真实执行回放；只看历史市场上模型方向是否对，同向时才复用旧真实成交盈亏。',
                'setHash': stable_hash(chosen[['marketSlug','pred_up15','actualUp']].to_dict('records')),
            })
        return rows, scan_audit
    except Exception as exc:
        return [{'scope':'archive_error','error':repr(exc)[:500]}], {'error': repr(exc)[:500]}


def train_quality_scores(train: pd.DataFrame, test: pd.DataFrame, features: list[str], label: str) -> np.ndarray:
    # Keep deliberately small/low-overfit. If lightgbm is absent, fall back to logistic.
    xtr = train[features].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    ytr = train[label].astype(int).to_numpy()
    xte = test[features].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    if len(train) < 500 or len(np.unique(ytr)) < 2:
        return np.full(len(test), 0.5)
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.035, num_leaves=15, min_child_samples=80, subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0, random_state=RNG_SEED, n_jobs=1, verbose=-1)
        model.fit(xtr, ytr)
        return np.asarray(model.predict_proba(xte)[:,1], dtype=float)
    except Exception:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.5, random_state=RNG_SEED))
        model.fit(xtr, ytr)
        return np.asarray(model.predict_proba(xte)[:,1], dtype=float)


def quality_model_methods(df: pd.DataFrame, window: str) -> list[MetricRow]:
    end = df['dt'].max()
    days = 180 if window == '180d' else 365
    start = end - pd.Timedelta(days=days)
    val = df[df['dt'] >= start].copy().sort_values('dt')
    train = df[df['dt'] < start - pd.Timedelta(hours=4)].copy().sort_values('dt')
    # Features available before trade only. Explicitly exclude proxy fill outcome/path labels.
    feature_cols = ['score15','proxy_visible_price','proxy_entry_price','ob_spread_ratio_last','ob_spread_ratio_mean','ob_depth_imbalance_last','ob_bid_ask_imbalance_last','ob_pressure_last']
    feature_cols = [c for c in feature_cols if c in df.columns]
    out=[]
    base_mask = val['proxy_visible_price'].notna()
    out.append(metric_from_proxy(val, '当前信号_代理吃价', window, base_mask, 'proxy_visible_price', None, '弱真相：token路径首价，保证成交代理。'))
    # Fixed transparent gates.
    for px in [0.52, 0.55, 0.60, 0.70]:
        mask = base_mask & (pd.to_numeric(val['proxy_visible_price'], errors='coerce') <= px)
        out.append(metric_from_proxy(val, f'固定买价门<= {px:.2f}', window, mask, 'proxy_visible_price', None, '弱真相：只用可见买价过滤，检验高买价是否伤害曲线。'))
    # Execution/quality model gates: labels are proxy-only, so outputs are marked weak.
    train2 = train.dropna(subset=['proxy_visible_price']).copy()
    val2 = val.copy()
    if len(train2) >= 500 and feature_cols:
        train2['label_win'] = train2['won'].astype(int)
        train2['label_good_exec'] = ((train2['won'].astype(bool)) & (pd.to_numeric(train2['proxy_visible_price'], errors='coerce') <= 0.60)).astype(int)
        p_win = train_quality_scores(train2, val2, feature_cols, 'label_win')
        p_good = train_quality_scores(train2, val2, feature_cols, 'label_good_exec')
        val2['quality_p_win'] = p_win
        val2['quality_p_good_exec'] = p_good
        val2['quality_ev'] = p_win - pd.to_numeric(val2['proxy_visible_price'], errors='coerce').fillna(1.0)
        for margin in [0.00, 0.02, 0.04, 0.06]:
            mask = base_mask & (val2['quality_ev'] >= margin)
            out.append(metric_from_proxy(val2, f'价值模型_EV>={margin:.2f}', window, mask, 'proxy_visible_price', None, '弱真相：模型胜率减可见买价，只作研究，不当真钱结论。'))
        for thr in [0.52, 0.56, 0.60]:
            mask = base_mask & (val2['quality_p_good_exec'] >= thr)
            out.append(metric_from_proxy(val2, f'成交质量模型_p>={thr:.2f}', window, mask, 'proxy_visible_price', None, '弱真相：用代理标签预测好成交，必须再经真实盘口验证。'))
    return out


def official_dryrun_rows(dry: pd.DataFrame) -> list[dict[str, Any]]:
    if dry.empty:
        return []
    rows=[]
    # One row per observed quote, and one row per market best eligible observation.
    d = dry.copy()
    d['is_valid_quote'] = d['best_ask'].notna() & d['spread'].notna()
    d['depth_1pct_850'] = d['ask_depth'].fillna(0) * d['best_ask'].fillna(0) >= START_BANKROLL * STAKE_FRAC
    for name, mask in [
        ('全部官方dryrun报价', d['is_valid_quote']),
        ('入场窗口内报价', d['is_valid_quote'] & d['entry_allowed']),
        ('入场窗口+深度覆盖1%', d['is_valid_quote'] & d['entry_allowed'] & d['depth_1pct_850']),
    ]:
        sub_all = d[mask].copy()
        if sub_all.empty:
            rows.append({'scope': name, 'quotes': int(mask.sum()), 'markets': 0, 'knownOutcomeMarkets': 0})
            continue
        # Deduplicate market by earliest quote in that scope.
        first_all = sub_all.sort_values('observed_at').drop_duplicates('market_slug', keep='first')
        first = first_all.dropna(subset=['won']).copy()
        wins = int(first['won'].astype(bool).sum()) if len(first) else 0
        losses = int(len(first)-wins)
        rows.append({
            'scope': name,
            'truthTier': 'medium_official_current_orderbook_dryrun',
            'quotes': int(mask.sum()),
            'markets': int(first_all['market_slug'].nunique()),
            'knownOutcomeMarkets': int(first['market_slug'].nunique()) if len(first) else 0,
            'wins': wins,
            'losses': losses,
            'winRatePct': round(100*wins/len(first),6) if len(first) else 0,
            'avgAsk': round(float(first_all['best_ask'].mean()),6),
            'avgSpread': round(float(first_all['spread'].mean()),6),
            'avgAskDepth': round(float(first_all['ask_depth'].mean()),6),
            'winnerAvgAsk': round(float(first[first['won'].astype(bool)]['best_ask'].mean()),6) if wins else 0,
            'loserAvgAsk': round(float(first[~first['won'].astype(bool)]['best_ask'].mean()),6) if losses else 0,
            'setHash': stable_hash(first_all[['market_slug','side','won','best_ask']].to_dict('records')),
        })
    return rows


def data_truth_report(candidates: pd.DataFrame, proxy: pd.DataFrame, dry: pd.DataFrame, live: pd.DataFrame, archive_rows: list[dict[str, Any]], archive_audit: dict[str, Any]) -> dict[str, Any]:
    local_ob_rows = 0; local_ob_markets = 0; local_depth_rows = 0
    if LOCAL_ORDERBOOK.exists():
        markets=set()
        with LOCAL_ORDERBOOK.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d=json.loads(line)
                except Exception:
                    continue
                local_ob_rows += 1
                if d.get('market_slug'):
                    markets.add(d.get('market_slug'))
                if d.get('ask_depth_top3') is not None or d.get('asks_top'):
                    local_depth_rows += 1
        local_ob_markets=len(markets)
    payload = {
        'generatedAt': now_iso(),
        'beijingTime': bj_now(),
        'liveConfigMutated': False,
        'truthVerdict': 'old P50/FAK+2 reports are not live-trading proof; this report separates strong/medium/weak truth tiers',
        'strongTruth': {
            'top159OfficialMatchedRows': int(len(live)),
            'canaryLedger': str(CANARY_LEDGER),
            'archivedRealRows': archive_rows,
            'archiveAudit': archive_audit,
        },
        'mediumTruth': {
            'officialDryrunRows': int(len(dry)),
            'officialDryrunMarkets': int(dry['market_slug'].nunique()) if not dry.empty else 0,
            'officialDryrunPath': str(OFFICIAL_DRYRUN),
            'localOrderbookRows': local_ob_rows,
            'localOrderbookMarkets': local_ob_markets,
            'localOrderbookRowsWithDepthOrLevels': local_depth_rows,
            'localOrderbookDirectionMapping': 'not reliable enough for historical directional FAK replay; used only for coverage and future data plan',
        },
        'weakTruth': {
            'candidateRows': int(len(candidates)),
            'proxyJoinedRows': int(len(proxy)),
            'proxyRowsWithVisiblePrice': int(proxy['proxy_visible_price'].notna().sum()),
            'episodePath': str(EPISODES),
            'proxyLimit': 'token path and proxy fill fields are research labels/features only; they are not proof of official fills',
        },
        'auditFixes': {
            'randomLabelTargetEncodingLeak': 'identified in previous shock-aware script; execution-quality script uses no target encoding',
            'fak2FullFillProblem': 'deprecated as live decision metric; full-fill proxy remains weak-tier only if shown',
            'archivedReplayProblem': 'archived real windows are reported as sanity checks, not as new-model execution replay',
        },
    }
    return payload


def markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    lines=['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']
    for r in rows:
        lines.append('|'+'|'.join(str(r.get(h,'')) for h in headers)+'|')
    return '\n'.join(lines)


def render_data_md(payload: dict[str, Any], dry_rows: list[dict[str, Any]]) -> str:
    lines=['# top159 成交质量/盘口模型：数据真相', '']
    lines.append(f"- 北京时间：`{payload['beijingTime']}`")
    lines.append(f"- 结论：`{payload['truthVerdict']}`")
    lines.append('- 强真相：官网确认成交、旧真实成交归档。')
    lines.append('- 中真相：官方当前盘口 dry-run、本地历史订单簿覆盖，但历史方向映射不足。')
    lines.append('- 弱真相：token 路径和原始K线代理，只能研究，不能当真钱收益。')
    lines.append('')
    lines.append('## 官方 dry-run 盘口摘要')
    lines.append(markdown_table(dry_rows, ['scope','truthTier','quotes','markets','knownOutcomeMarkets','wins','losses','winRatePct','avgAsk','avgSpread','avgAskDepth','winnerAvgAsk','loserAvgAsk','setHash']))
    lines.append('')
    lines.append('## 历史归档真实交易摘要')
    lines.append(markdown_table(payload['strongTruth']['archivedRealRows'], ['scope','oldRealMarkets','selectedTrades','wins','losses','winRatePct','oneUnitPnl','fak52Pnl','sameDirectionActualPnlUsd','truthLimit','setHash']))
    lines.append('')
    lines.append('## 数据覆盖')
    lines.append('```json')
    lines.append(json.dumps({k:payload[k] for k in ['strongTruth','mediumTruth','weakTruth','auditFixes']}, ensure_ascii=False, indent=2, default=str))
    lines.append('```')
    return '\n'.join(lines)+'\n'


def render_compare_md(rows: list[MetricRow], live_row: MetricRow, archive_rows: list[dict[str, Any]]) -> str:
    headers=['window','method','truth_tier','trades','executable','wins','losses','pending','win_rate_pct','pnl','ending_funds','max_drawdown','avg_buy_price','avg_spread_cost','winner_fill_rate_pct','loser_fill_rate_pct','winner_avg_fill_fraction','loser_avg_fill_fraction','official_missing','orderbook_missing','retention_pct','set_hash','note']
    lines=['# top159 成交质量/盘口模型：180天/365天对比', '']
    lines.append('## 强真相：当前 top159 官网成交')
    lines.append(markdown_table([asdict(live_row)], headers))
    lines.append('')
    lines.append('## 弱真相：180天/365天代理研究表')
    lines.append(markdown_table([asdict(r) for r in rows], headers))
    lines.append('')
    lines.append('## 历史归档真实交易 sanity check')
    lines.append(markdown_table(archive_rows, ['scope','oldRealMarkets','selectedTrades','wins','losses','winRatePct','oneUnitPnl','fak52Pnl','sameDirectionActualPnlUsd','truthLimit','setHash']))
    lines.append('')
    lines.append('## 口径提醒')
    lines.append('- `weak_token_path_proxy` 不能证明真钱能成交，只能比较候选过滤是否可能改善。')
    lines.append('- `strong_official_fills` 只统计已经官网成交的 top159 真钱单，样本少但真。')
    lines.append('- 历史归档表不是新模型真实回放，只能看旧市场上方向是否更差。')
    return '\n'.join(lines)+'\n'


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    candidates, model_truth = build_current_live_candidates()
    proxy = join_proxy_features(candidates)
    live_fills = parse_live_official_fills()
    dry = parse_official_dryrun()
    archive_rows, archive_audit = archived_rows_for_candidate(candidates)
    dry_rows = official_dryrun_rows(dry)

    data_payload = data_truth_report(candidates, proxy, dry, live_fills, archive_rows, archive_audit)
    data_payload['currentModelTruth'] = model_truth
    write_json(OUT_DATA_JSON, data_payload)
    write_text(OUT_DATA_MD, render_data_md(data_payload, dry_rows))

    metric_rows: list[MetricRow] = []
    for window in ['180d','365d']:
        end = proxy['dt'].max()
        start = end - pd.Timedelta(days=180 if window=='180d' else 365)
        wdf = proxy[proxy['dt'] >= start].copy().sort_values('dt')
        metric_rows.extend(quality_model_methods(wdf, window))

    live_row = summarize_live_fills(live_fills)
    compare_payload = {
        'generatedAt': now_iso(),
        'beijingTime': bj_now(),
        'liveConfigMutated': False,
        'oldReportsCredibilityVerdict': {
            'p50': 'not live PnL; only slot1-toxic proxy',
            'fak2': 'optimistic full-fill proxy; not official FAK execution proof',
            'archivedReal': 'sanity check only; not new strategy replay',
            'randomLabelAudit': 'previous target-encoding random label audit has a known flaw; this script uses no target encoding',
        },
        'liveOfficialFillMetric': asdict(live_row),
        'proxyMetrics': [asdict(r) for r in metric_rows],
        'officialDryrunRows': dry_rows,
        'archivedRealRows': archive_rows,
    }
    write_json(OUT_COMPARE_JSON, compare_payload)
    write_text(OUT_COMPARE_MD, render_compare_md(metric_rows, live_row, archive_rows))

    # Decision: find candidates that are not obviously worse in proxy and do not hurt archived sanity.
    proxy_candidates = [r for r in metric_rows if r.truth_tier == 'weak_token_path_proxy' and r.window in {'180d','365d'}]
    by_method: dict[str, list[MetricRow]] = {}
    for r in proxy_candidates:
        by_method.setdefault(r.method, []).append(r)
    viable=[]
    for method, rs in by_method.items():
        if len(rs) != 2:
            continue
        if all(r.executable >= (100 if r.window=='180d' else 200) for r in rs) and all(r.retention_pct >= 45.0 for r in rs) and all(r.ending_funds > START_BANKROLL for r in rs):
            # Do not rank by explosive pnl only; penalize loser fill > winner fill.
            score = sum(r.ending_funds - START_BANKROLL for r in rs) - sum(max(0.0, r.loser_fill_rate_pct - r.winner_fill_rate_pct) * 5.0 for r in rs) - sum(r.max_drawdown * 0.2 for r in rs)
            viable.append({'method': method, 'score': round(score,6), 'rows': [asdict(r) for r in rs]})
    viable.sort(key=lambda x:x['score'], reverse=True)
    verdict = {
        'generatedAt': now_iso(),
        'beijingTime': bj_now(),
        'status': 'research_only_no_live_switch',
        'credibilityVerdict': 'not credible enough for live switch; useful only after truth-tier separation',
        'selectedResearchCandidate': viable[0] if viable else None,
        'viableCount': len(viable),
        'nextAction': 'if user continues, run shadow collection with official orderbook and train execution-quality model only on strong/medium truth; do not switch live from weak proxy alone',
    }
    write_json(OUT_VERDICT_JSON, verdict)
    lines=['# top159 成交质量/盘口模型：唯一结论', '']
    lines.append(f"- 北京时间：`{verdict['beijingTime']}`")
    lines.append(f"- 可信度结论：`{verdict['credibilityVerdict']}`")
    if viable:
        lines.append(f"- 研究候选：`{viable[0]['method']}`，但它仍是弱真相代理，不能直接切真钱。")
    else:
        lines.append('- 没有候选同时满足 180/365 的基础交易数、保留率和期末资金门槛。')
    lines.append('- 真实可上线前必须先增加官方盘口影子样本；不能再用 P50/FAK+2 直接决策。')
    lines.append('')
    lines.append('## 输出文件')
    for p in [OUT_DATA_MD, OUT_COMPARE_MD, OUT_VERDICT_MD, OUT_DATA_JSON, OUT_COMPARE_JSON, OUT_VERDICT_JSON]:
        lines.append(f'- `{p}`')
    write_text(OUT_VERDICT_MD, '\n'.join(lines)+'\n')

    print(OUT_DATA_MD)
    print(OUT_COMPARE_MD)
    print(OUT_VERDICT_MD)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
