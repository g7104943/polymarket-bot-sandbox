#!/usr/bin/env python3
"""Research-only 061 Bayesian dynamic sizer V2.

This script does not touch live trading config. It evaluates dynamic stake sizing
on top of the current 061 signal stream using only time-ordered past data.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import random
import sys
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path('/Users/mac/polyfun')
SCRIPTS = ROOT / 'polyfun-next' / 'scripts'
REPORTS = ROOT / 'reports'
REPORTS.mkdir(parents=True, exist_ok=True)

V1_PATH = SCRIPTS / 'run_top159_061_bayesian_sizer_search.py'


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load module: {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


v1 = _load_module(V1_PATH, 'bayes_v1')
base = v1.base
actual = v1.actual
shock = v1.shock

BUY_PRICE = 0.55
INITIAL_FUNDS = 850.0
STAKE_BASE = 0.01
CURRENT_061_PARAMS = v1.base.base.CURRENT_061_PARAMS
BASELINE_061 = v1.BASELINE_061

# Baseline official replay from V1 report. We recompute it in this script too,
# but keeping names explicit makes the final table less ambiguous.


@dataclass(frozen=True)
class V2Policy:
    name: str
    dims: Tuple[str, ...]
    prior_strength: float
    min_count: float
    lookback_days: Optional[int]
    kelly_fraction: float
    min_stake: float
    max_stake: float
    edge_buffer: float
    allow_skip: bool
    skip_prob: float
    weak_window: int
    weak_win_min: float
    weak_cap: float
    weak_dd: float
    update_lag: int
    use_actual_price: bool = False


@dataclass
class WindowMetrics:
    rows: int
    trades: int
    skipped: int
    wins: int
    losses: int
    winRate: float
    avgStakePct: float
    avgStakeMultiplier: float
    compoundPnl: float
    endingFunds: float
    maxDrawdown: float
    returnDrawdown: float
    downsizeCount: int
    upsizeCount: int
    strongUpsizeCount: int
    weakRegimeCount: int
    setHash: str
    originalLiveAmountPnl: Optional[float] = None
    actualPriceDynamicPnl: Optional[float] = None
    weightedAvgBuyPrice: Optional[float] = None


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n')


def pct(x: float) -> str:
    return f'{x * 100:.2f}%'


def money(x: float) -> str:
    return f'{x:,.2f}'


def digest_rows(parts: Iterable[Any]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode('utf-8'))
        h.update(b'|')
    return h.hexdigest()[:16]


def prepare_data() -> Dict[str, Any]:
    enriched, atom_store, period_vals = base.build_truth()

    train180 = v1.add_bucket_features(v1.current_061_selected(period_vals, atom_store, "gate_train_for_180d"))
    val180 = v1.add_bucket_features(v1.current_061_selected(period_vals, atom_store, "validation_180d"))
    train365 = v1.add_bucket_features(v1.current_061_selected(period_vals, atom_store, "gate_train_for_365d"))
    val365 = v1.add_bucket_features(v1.current_061_selected(period_vals, atom_store, "validation_365d"))

    official_rows = actual.load_actual_logical_rows(strict_061=False)
    if not official_rows.empty:
        min_dt = pd.to_datetime(official_rows["dt"], utc=True).min()
        hist = enriched[pd.to_datetime(enriched["dt"], utc=True) < min_dt].sort_values("dt").drop_duplicates("dt", keep="last").reset_index(drop=True)
        hist = shock.enrich_shock_features(hist)
        hist_atom = v1.base.base.atom_masks(hist)
        condition = v1.base.base.condition_for_candidate({"h": hist_atom}, "h", CURRENT_061_PARAMS)
        score = pd.to_numeric(hist["score15"], errors="coerce").fillna(0.0).to_numpy()
        official_train = v1.add_bucket_features(hist[((~condition) | (score >= float(CURRENT_061_PARAMS["shock_score_min"])))].copy().reset_index(drop=True))
        cols = ["dt", "label_up", "pred_up15", "direction", "score15", "won", "period_name", "market_slug", "actual_cost", "actual_shares", "actual_avg_price", "actual_pnl"]
        off = shock.enrich_shock_features(official_rows[[c for c in cols if c in official_rows.columns]].copy())
        official_rows = v1.add_bucket_features(off.sort_values("dt").reset_index(drop=True))
    else:
        official_train = pd.DataFrame()

    return {
        'train180': train180.reset_index(drop=True),
        'val180': val180.reset_index(drop=True),
        'train365': train365.reset_index(drop=True),
        'val365': val365.reset_index(drop=True),
        'official_train': official_train.reset_index(drop=True),
        'official_rows': official_rows.reset_index(drop=True),
    }


def _ensure_cols(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 'NA'
    return out


def hierarchy_levels(dims: Tuple[str, ...]) -> List[Tuple[str, ...]]:
    levels: List[Tuple[str, ...]] = [tuple()]
    if 'direction' in dims:
        levels.append(('direction',))
    if 'score_bin' in dims:
        levels.append(('score_bin',))
    if 'score_bin' in dims and 'direction' in dims:
        levels.append(('score_bin', 'direction'))
    # Add partial environment level if available.
    env = tuple([d for d in dims if d not in {'score_bin', 'direction'}][:2])
    if env:
        levels.append(env)
    if dims not in levels:
        levels.append(dims)
    # Deduplicate preserving order.
    seen = set()
    uniq = []
    for lev in levels:
        if lev not in seen:
            seen.add(lev)
            uniq.append(lev)
    return uniq


class HierarchicalCounter:
    def __init__(self, levels: List[Tuple[str, ...]], prior_strength: float, min_count: float, lookback_days: Optional[int]):
        self.levels = levels
        self.prior_strength = prior_strength
        self.min_count = min_count
        self.lookback_ns = None if lookback_days is None else int(lookback_days * 24 * 3600 * 1_000_000_000)
        self.events: List[Dict[Tuple[Any, ...], deque]] = [defaultdict(deque) for _ in levels]
        self.static_counts: List[Dict[Tuple[Any, ...], List[float]]] = [defaultdict(lambda: [0.0, 0.0]) for _ in levels]

    @staticmethod
    def key(row: pd.Series, dims: Tuple[str, ...]) -> Tuple[Any, ...]:
        if not dims:
            return ('__global__',)
        return tuple(row.get(d, 'NA') for d in dims)

    def _pruned_counts(self, level_i: int, key: Tuple[Any, ...], now_ns: int) -> Tuple[float, float]:
        if self.lookback_ns is None:
            wins, n = self.static_counts[level_i].get(key, [0.0, 0.0])
            return float(wins), float(n)
        q = self.events[level_i].get(key)
        if not q:
            return 0.0, 0.0
        cutoff = now_ns - self.lookback_ns
        while q and q[0][0] < cutoff:
            q.popleft()
        if not q:
            return 0.0, 0.0
        wins = sum(1.0 for _, won in q if won)
        return wins, float(len(q))

    def estimate(self, row: pd.Series, now_ns: int, global_prior: float) -> Tuple[float, float, Tuple[str, ...]]:
        parent_p = global_prior
        used_n = 0.0
        used_level: Tuple[str, ...] = tuple()
        for i, dims in enumerate(self.levels):
            k = self.key(row, dims)
            wins, n = self._pruned_counts(i, k, now_ns)
            p = (wins + self.prior_strength * parent_p) / (n + self.prior_strength) if n + self.prior_strength > 0 else parent_p
            if not dims or n >= self.min_count:
                parent_p = p
                used_n = n
                used_level = dims
        return float(parent_p), float(used_n), used_level

    def add(self, row: pd.Series, ts_ns: int, won: bool) -> None:
        for i, dims in enumerate(self.levels):
            k = self.key(row, dims)
            if self.lookback_ns is None:
                self.static_counts[i][k][0] += 1.0 if won else 0.0
                self.static_counts[i][k][1] += 1.0
            else:
                self.events[i][k].append((ts_ns, bool(won)))


def row_ts_ns(row: pd.Series) -> int:
    ts = pd.Timestamp(row['dt'])
    if ts.tzinfo is not None:
        ts = ts.tz_convert('UTC').tz_localize(None)
    return int(ts.value)


def global_win_rate(train: pd.DataFrame) -> float:
    if train.empty:
        return 0.5
    return float(train['won'].mean())


def stake_from_p(policy: V2Policy, p: float, buy_price: float, recent_weak: bool) -> Tuple[float, str]:
    b = (1.0 / buy_price) - 1.0
    # Probability must clear the price break-even plus optional buffer to avoid
    # increasing stake on barely positive, noisy buckets.
    break_even = buy_price + policy.edge_buffer
    if policy.allow_skip and p < policy.skip_prob:
        return 0.0, 'skip'
    if b <= 0:
        raw = STAKE_BASE
    else:
        kelly = (b * p - (1.0 - p)) / b
        raw = policy.kelly_fraction * kelly
    if raw <= 0:
        raw = policy.min_stake
    stake = min(max(raw, policy.min_stake), policy.max_stake)
    if p < break_even:
        stake = min(stake, STAKE_BASE * 0.75)
    if recent_weak:
        stake = min(stake, policy.weak_cap)
    if stake <= 0:
        return 0.0, 'skip'
    if stake < STAKE_BASE * 0.9:
        return stake, 'downsize'
    if stake > STAKE_BASE * 1.4:
        return stake, 'strong_upsize'
    if stake > STAKE_BASE * 1.05:
        return stake, 'upsize'
    return stake, 'base'


def simulate_v2(train: pd.DataFrame, val: pd.DataFrame, policy: V2Policy, *, buy_price: float = BUY_PRICE, actual_mode: bool = False) -> WindowMetrics:
    dims = tuple(policy.dims)
    needed = sorted(set(dims + ('dt', 'won', 'direction', 'score_bin')))
    train = _ensure_cols(train, needed).sort_values('dt').reset_index(drop=True)
    val = _ensure_cols(val, needed).sort_values('dt').reset_index(drop=True)
    levels = hierarchy_levels(dims)
    prior = global_win_rate(train)
    counter = HierarchicalCounter(levels, policy.prior_strength, policy.min_count, policy.lookback_days)

    for _, row in train.iterrows():
        counter.add(row, row_ts_ns(row), bool(row['won']))

    funds = INITIAL_FUNDS
    peak = funds
    max_dd = 0.0
    trades = skipped = wins = losses = 0
    down = up = strong = weak_n = 0
    stake_pcts: List[float] = []
    hash_parts: List[Any] = []
    pending: deque = deque()
    recent: deque = deque(maxlen=max(policy.weak_window, 1) if policy.weak_window > 0 else 1)
    recent_peak = funds
    original_live_amount_pnl = 0.0 if actual_mode else None
    weighted_cost = 0.0
    weighted_amt = 0.0

    def flush_until(idx: int):
        nonlocal recent_peak
        while pending and pending[0][0] <= idx:
            _, prow = pending.popleft()
            counter.add(prow, row_ts_ns(prow), bool(prow['won']))
            if policy.weak_window > 0:
                recent.append(1 if bool(prow['won']) else 0)
        recent_peak = max(recent_peak, funds)

    for i, row in val.iterrows():
        flush_until(i)
        now_ns = row_ts_ns(row)
        p, eff_n, used_level = counter.estimate(row, now_ns, prior)

        recent_weak = False
        if policy.weak_window > 0 and len(recent) >= policy.weak_window:
            if float(np.mean(recent)) < policy.weak_win_min:
                recent_weak = True
            if recent_peak > 0 and (recent_peak - funds) / recent_peak >= policy.weak_dd:
                recent_weak = True
        if recent_weak:
            weak_n += 1

        q = buy_price
        if actual_mode and policy.use_actual_price and 'actual_avg_price' in row and pd.notna(row['actual_avg_price']):
            q = float(row['actual_avg_price'])
            # Protect against malformed official/cached prices.
            if not (0.01 <= q <= 0.99):
                q = buy_price
        stake_pct, action = stake_from_p(policy, p, q, recent_weak)

        pending.append((i + policy.update_lag, row))
        if stake_pct <= 0:
            skipped += 1
            hash_parts.append((row.get('market_slug', row.get('market', i)), 'skip', round(p, 6), tuple(used_level)))
            continue

        stake = funds * stake_pct
        won = bool(row['won'])
        if won:
            pnl = stake * ((1.0 / q) - 1.0)
            wins += 1
        else:
            pnl = -stake
            losses += 1
        funds += pnl
        peak = max(peak, funds)
        max_dd = max(max_dd, peak - funds)
        trades += 1
        stake_pcts.append(stake_pct)
        if action == 'downsize':
            down += 1
        elif action == 'upsize':
            up += 1
        elif action == 'strong_upsize':
            strong += 1
        if actual_mode:
            if original_live_amount_pnl is not None:
                original_live_amount_pnl += float(row.get('actual_pnl', 0.0) or 0.0)
            amt = float(row.get('actual_cost', 0.0) or 0.0)
            if amt > 0 and 'actual_avg_price' in row and pd.notna(row['actual_avg_price']):
                weighted_cost += amt * float(row['actual_avg_price'])
                weighted_amt += amt
        hash_parts.append((row.get('market_slug', row.get('market', i)), action, round(p, 6), round(stake_pct, 6), won, round(q, 5), tuple(used_level)))

    wr = wins / trades if trades else 0.0
    avg_stake = float(np.mean(stake_pcts)) if stake_pcts else 0.0
    pnl = funds - INITIAL_FUNDS
    return WindowMetrics(
        rows=int(len(val)),
        trades=int(trades),
        skipped=int(skipped),
        wins=int(wins),
        losses=int(losses),
        winRate=float(wr),
        avgStakePct=avg_stake,
        avgStakeMultiplier=avg_stake / STAKE_BASE if STAKE_BASE else 0.0,
        compoundPnl=float(pnl),
        endingFunds=float(funds),
        maxDrawdown=float(max_dd),
        returnDrawdown=float(pnl / max_dd) if max_dd > 0 else 0.0,
        downsizeCount=int(down),
        upsizeCount=int(up),
        strongUpsizeCount=int(strong),
        weakRegimeCount=int(weak_n),
        setHash=digest_rows(hash_parts),
        originalLiveAmountPnl=float(original_live_amount_pnl) if original_live_amount_pnl is not None else None,
        actualPriceDynamicPnl=float(pnl) if actual_mode else None,
        weightedAvgBuyPrice=float(weighted_cost / weighted_amt) if weighted_amt > 0 else None,
    )


def fixed_metrics(df: pd.DataFrame, *, buy_price: float = BUY_PRICE, actual_mode: bool = False) -> WindowMetrics:
    funds = INITIAL_FUNDS
    peak = funds
    max_dd = 0.0
    wins = losses = 0
    hash_parts = []
    original_live_amount_pnl = 0.0 if actual_mode else None
    weighted_cost = 0.0
    weighted_amt = 0.0
    for i, row in df.sort_values('dt').reset_index(drop=True).iterrows():
        q = buy_price
        if actual_mode and 'actual_avg_price' in row and pd.notna(row['actual_avg_price']):
            q = float(row['actual_avg_price'])
            if not (0.01 <= q <= 0.99):
                q = buy_price
        stake = funds * STAKE_BASE
        won = bool(row['won'])
        if won:
            pnl = stake * ((1.0 / q) - 1.0)
            wins += 1
        else:
            pnl = -stake
            losses += 1
        funds += pnl
        peak = max(peak, funds)
        max_dd = max(max_dd, peak - funds)
        if actual_mode:
            if original_live_amount_pnl is not None:
                original_live_amount_pnl += float(row.get('actual_pnl', 0.0) or 0.0)
            amt = float(row.get('actual_cost', 0.0) or 0.0)
            if amt > 0 and 'actual_avg_price' in row and pd.notna(row['actual_avg_price']):
                weighted_cost += amt * float(row['actual_avg_price'])
                weighted_amt += amt
        hash_parts.append((row.get('market_slug', row.get('market', i)), 'fixed', won, round(q, 5)))
    trades = wins + losses
    pnl = funds - INITIAL_FUNDS
    return WindowMetrics(
        rows=int(len(df)), trades=int(trades), skipped=0, wins=int(wins), losses=int(losses),
        winRate=float(wins / trades if trades else 0.0), avgStakePct=STAKE_BASE, avgStakeMultiplier=1.0,
        compoundPnl=float(pnl), endingFunds=float(funds), maxDrawdown=float(max_dd),
        returnDrawdown=float(pnl / max_dd) if max_dd else 0.0,
        downsizeCount=0, upsizeCount=0, strongUpsizeCount=0, weakRegimeCount=0,
        setHash=digest_rows(hash_parts),
        originalLiveAmountPnl=float(original_live_amount_pnl) if original_live_amount_pnl is not None else None,
        actualPriceDynamicPnl=float(pnl) if actual_mode else None,
        weightedAvgBuyPrice=float(weighted_cost / weighted_amt) if weighted_amt > 0 else None,
    )


def policy_grid() -> List[V2Policy]:
    # Compact high-information V2 grid. The wide grid is too slow in pure
    # Python and adds mostly redundant combinations. This grid keeps every
    # conceptual branch: hierarchy depth, no/30/60 day memory, conservative/
    # medium Kelly, drawdown control, and a tiny skip family.
    dims_options = [
        ('score_bin', 'direction'),
        ('score_bin', 'direction', '1h_vol_bucket', '4h_vol_bucket'),
        ('score_bin', 'direction', '1h_trend_bucket', '4h_pos_bucket'),
        ('score_bin', 'direction', 'hour_bucket', '1h_vol_bucket'),
    ]
    policies: List[V2Policy] = []
    for dims in dims_options:
        for prior in [200.0, 400.0]:
            for min_count in [60.0, 120.0]:
                for lookback in [None, 30, 60]:
                    for kf in [0.20, 0.35]:
                        for max_stake in [0.015, 0.018]:
                            for weak_window in [0, 50]:
                                if weak_window == 0:
                                    weak_win_min = 0.0; weak_cap = max_stake; weak_dd = 1.0
                                else:
                                    weak_win_min = 0.535; weak_cap = 0.0075; weak_dd = 0.08
                                nm = digest_rows([dims, prior, min_count, lookback, kf, max_stake, weak_window])
                                policies.append(V2Policy(
                                    name=f'v2_{nm}', dims=tuple(dims), prior_strength=prior,
                                    min_count=min_count, lookback_days=lookback, kelly_fraction=kf,
                                    min_stake=0.0035, max_stake=max_stake, edge_buffer=0.0,
                                    allow_skip=False, skip_prob=0.0, weak_window=weak_window,
                                    weak_win_min=weak_win_min, weak_cap=weak_cap, weak_dd=weak_dd, update_lag=2,
                                ))
    for dims in [('score_bin', 'direction', '1h_vol_bucket', '4h_vol_bucket'), ('score_bin', 'direction', 'hour_bucket', '1h_trend_bucket')]:
        for lookback in [None, 60]:
            for skip_prob in [0.525, 0.535]:
                nm = digest_rows(['skip', dims, lookback, skip_prob])
                policies.append(V2Policy(
                    name=f'v2_skip_{nm}', dims=tuple(dims), prior_strength=300.0, min_count=120.0,
                    lookback_days=lookback, kelly_fraction=0.25, min_stake=0.0035, max_stake=0.015,
                    edge_buffer=0.0, allow_skip=True, skip_prob=skip_prob, weak_window=50,
                    weak_win_min=0.535, weak_cap=0.0075, weak_dd=0.08, update_lag=2,
                ))
    return policies

def pass_mid_gate(m180: WindowMetrics, m365: WindowMetrics, off: Optional[WindowMetrics], b180: WindowMetrics, b365: WindowMetrics, boff: Optional[WindowMetrics]) -> Tuple[bool, List[str]]:
    reasons = []
    if m180.compoundPnl < b180.compoundPnl * 1.10:
        reasons.append('180盈亏未高于061 10%')
    if m365.compoundPnl < b365.compoundPnl * 1.20:
        reasons.append('365盈亏未高于061 20%')
    if m180.maxDrawdown > b180.maxDrawdown * 1.05:
        reasons.append('180回撤高于061超过5%')
    if m365.maxDrawdown > b365.maxDrawdown * 1.05:
        reasons.append('365回撤高于061超过5%')
    retention = m365.trades / b365.trades if b365.trades else 0.0
    if retention < 0.85 and not (m365.compoundPnl >= b365.compoundPnl * 1.30 and m365.maxDrawdown < b365.maxDrawdown):
        reasons.append('365交易保留率不足85%且未满足30%收益+更低回撤例外')
    if off is not None and boff is not None:
        if off.actualPriceDynamicPnl is not None and boff.actualPriceDynamicPnl is not None:
            if off.actualPriceDynamicPnl < boff.actualPriceDynamicPnl:
                reasons.append('官方实际价格动态回放盈亏低于061')
        if off.maxDrawdown > boff.maxDrawdown * 1.10:
            reasons.append('官方实际价格动态回放回撤明显更差')
    return len(reasons) == 0, reasons


def random_label_audit(data: Dict[str, Any], sample_policies: List[V2Policy], true_b180: WindowMetrics, true_b365: WindowMetrics) -> Dict[str, Any]:
    # Stress test the whole calibration loop, not just the validation labels.
    # Both train and validation labels are shuffled, then candidates are judged
    # against the real 061 hurdle. A random world must not produce a deployable
    # middle-improvement candidate.
    total_strict = 0
    best_by_seed = []
    sampled = min(20, len(sample_policies))
    for seed in [12345, 23456, 34567]:
        rng = np.random.default_rng(seed)
        tr180 = data['train180'].copy(); tr365 = data['train365'].copy()
        val180 = data['val180'].copy(); val365 = data['val365'].copy()
        tr180['won'] = rng.permutation(tr180['won'].to_numpy())
        tr365['won'] = rng.permutation(tr365['won'].to_numpy())
        val180['won'] = rng.permutation(val180['won'].to_numpy())
        val365['won'] = rng.permutation(val365['won'].to_numpy())
        strict_pass = 0
        best_pnl = -1e18
        for pol in sample_policies[:sampled]:
            m180 = simulate_v2(tr180, val180, pol)
            m365 = simulate_v2(tr365, val365, pol)
            if (m180.compoundPnl >= true_b180.compoundPnl * 1.10 and
                    m365.compoundPnl >= true_b365.compoundPnl * 1.20 and
                    m180.maxDrawdown <= true_b180.maxDrawdown * 1.05 and
                    m365.maxDrawdown <= true_b365.maxDrawdown * 1.05):
                strict_pass += 1
            best_pnl = max(best_pnl, m365.compoundPnl)
        total_strict += strict_pass
        best_by_seed.append({'seed': seed, 'strictPassCount': strict_pass, 'bestRandom365Pnl': best_pnl})
    return {'passed': total_strict == 0, 'strictPassCount': total_strict, 'bySeed': best_by_seed, 'sampledPolicies': sampled}


def feature_shift_audit(data: Dict[str, Any], policy: V2Policy) -> Dict[str, Any]:
    val = data['val365'].copy()
    cols = [c for c in policy.dims if c in val.columns]
    for c in cols:
        val[c] = val[c].shift(1).fillna(val[c].iloc[0])
    normal = simulate_v2(data['train365'], data['val365'], policy)
    shifted = simulate_v2(data['train365'], val, policy)
    changed = normal.setHash != shifted.setHash or abs(normal.compoundPnl - shifted.compoundPnl) > 1e-9
    return {
        'passed': changed,
        'policy': policy.name,
        'shiftedColumns': cols,
        'normalHash': normal.setHash,
        'shiftedHash': shifted.setHash,
        'normal365Pnl': normal.compoundPnl,
        'shifted365Pnl': shifted.compoundPnl,
    }


def metrics_to_dict(m: WindowMetrics) -> Dict[str, Any]:
    return asdict(m)


def run() -> None:
    data = prepare_data()
    b180 = fixed_metrics(data['val180'])
    b365 = fixed_metrics(data['val365'])
    off_base = None
    if not data['official_rows'].empty:
        off_base = fixed_metrics(data['official_rows'], actual_mode=True)

    baseline_ok = (
        b180.trades == BASELINE_061['180d']['trades'] and
        round(b180.compoundPnl, 6) == round(BASELINE_061['180d']['compoundPnl'], 6) and
        b365.trades == BASELINE_061['365d']['trades'] and
        round(b365.compoundPnl, 6) == round(BASELINE_061['365d']['compoundPnl'], 6)
    )

    policies = policy_grid()
    audit: Dict[str, Any] = {
        'baselineReproductionPassed': baseline_ok,
        'baseline180': metrics_to_dict(b180),
        'baseline365': metrics_to_dict(b365),
        'officialBaseline': metrics_to_dict(off_base) if off_base else None,
        'policyCount': len(policies),
    }
    if not baseline_ok:
        audit['blocked'] = True
        write_json(REPORTS / 'top159_061_bayesian_sizer_v2_bug_audit_latest.json', audit)
        raise SystemExit('Baseline reproduction failed; blocked V2 search.')

    audit['randomLabel'] = random_label_audit(data, policies, b180, b365)
    audit['featureShift'] = feature_shift_audit(data, next(p for p in policies if '1h_vol_bucket' in p.dims and '4h_vol_bucket' in p.dims))
    # We require the random label gate, and record feature-shift as a warning if
    # a particular policy is insensitive; the final top policy also gets repeat checked.
    audit['passed'] = bool(audit['baselineReproductionPassed'] and audit['randomLabel']['passed'] and audit['featureShift']['passed'])
    write_json(REPORTS / 'top159_061_bayesian_sizer_v2_bug_audit_latest.json', audit)

    lines = [
        '# 061 贝叶斯动态仓位 V2 审计',
        '',
        f'- 061基线复现: {"通过" if baseline_ok else "失败"}',
        f'- 随机标签测试: {"通过" if audit["randomLabel"]["passed"] else "失败"}，严格通过数={audit["randomLabel"]["strictPassCount"]}',
        f'- 特征错位测试: {"通过" if audit["featureShift"]["passed"] else "失败"}',
        f'- 搜索候选数: {len(policies)}',
        '',
    ]
    (REPORTS / 'top159_061_bayesian_sizer_v2_bug_audit_latest.md').write_text('\n'.join(lines), encoding='utf-8')
    if not audit['passed']:
        raise SystemExit('Audit failed; blocked V2 search.')

    results: List[Dict[str, Any]] = []
    for idx, pol in enumerate(policies, 1):
        m180 = simulate_v2(data['train180'], data['val180'], pol)
        m365 = simulate_v2(data['train365'], data['val365'], pol)
        off = None
        passed, reasons = pass_mid_gate(m180, m365, off, b180, b365, None)
        improvement_score = (
            (m180.compoundPnl - b180.compoundPnl) / max(abs(b180.compoundPnl), 1) * 0.30 +
            (m365.compoundPnl - b365.compoundPnl) / max(abs(b365.compoundPnl), 1) * 0.50 +
            (b365.maxDrawdown - m365.maxDrawdown) / max(b365.maxDrawdown, 1) * 0.20
        )
        results.append({
            'rankScore': improvement_score,
            'strictPass': passed,
            'failReasons': reasons,
            'policy': asdict(pol),
            'candidateId': digest_rows([pol.name, m180.setHash, m365.setHash, off.setHash if off else 'no_off']),
            'metrics': {
                '180d': metrics_to_dict(m180),
                '365d': metrics_to_dict(m365),
                'officialActualReplay': metrics_to_dict(off) if off else None,
            },
        })
        if idx % 250 == 0:
            write_json(REPORTS / 'top159_061_bayesian_sizer_v2_leaderboard_latest.json', sorted(results, key=lambda r: r['rankScore'], reverse=True)[:200])

    ranked = sorted(results, key=lambda r: r['rankScore'], reverse=True)
    # Official replay is expensive and only meaningful for plausible candidates.
    if off_base is not None and not data['official_rows'].empty:
        for r in ranked[:80]:
            pol = V2Policy(**r['policy'])
            pol_off = copy.copy(pol)
            object.__setattr__(pol_off, 'use_actual_price', True)
            off = simulate_v2(data['official_train'], data['official_rows'], pol_off, actual_mode=True)
            r['metrics']['officialActualReplay'] = metrics_to_dict(off)
            passed, reasons = pass_mid_gate(WindowMetrics(**r['metrics']['180d']), WindowMetrics(**r['metrics']['365d']), off, b180, b365, off_base)
            r['strictPass'] = passed
            r['failReasons'] = reasons
            r['rankScore'] += ((off.actualPriceDynamicPnl - off_base.actualPriceDynamicPnl) / max(abs(off_base.actualPriceDynamicPnl), 1) * 0.10 if off.actualPriceDynamicPnl is not None and off_base.actualPriceDynamicPnl is not None else 0.0)
        ranked = sorted(results, key=lambda r: r['rankScore'], reverse=True)
    # Repeat top 10 checks.
    repeat = []
    for r in ranked[:10]:
        pol = V2Policy(**r['policy'])
        a180 = simulate_v2(data['train180'], data['val180'], pol)
        a365 = simulate_v2(data['train365'], data['val365'], pol)
        repeat.append({
            'candidateId': r['candidateId'],
            'passed': a180.setHash == r['metrics']['180d']['setHash'] and a365.setHash == r['metrics']['365d']['setHash'],
            'repeat180Hash': a180.setHash,
            'repeat365Hash': a365.setHash,
        })
    strict = [r for r in ranked if r['strictPass']]
    unique = {
        'baseline': {'180d': metrics_to_dict(b180), '365d': metrics_to_dict(b365), 'officialActualReplay': metrics_to_dict(off_base) if off_base else None},
        'bestOverall': ranked[0] if ranked else None,
        'bestStrictPass': strict[0] if strict else None,
        'strictPassCount': len(strict),
        'repeatChecks': repeat,
        'auditPassed': audit['passed'],
        'decision': None,
    }
    if strict:
        unique['decision'] = 'FOUND_MID_IMPROVEMENT_CANDIDATE_FOR_SHADOW_ONLY'
    else:
        unique['decision'] = 'NO_MID_IMPROVEMENT_CANDIDATE; V2 only observation, do not switch live'

    write_json(REPORTS / 'top159_061_bayesian_sizer_v2_leaderboard_latest.json', ranked[:500])
    write_json(REPORTS / 'top159_061_bayesian_sizer_v2_unique_verdict_latest.json', unique)

    # Markdown compare.
    rows = []
    def add_row(name: str, window: str, m: WindowMetrics):
        rows.append([name, window, str(m.trades), f'{m.wins}/{m.losses}', pct(m.winRate), f'{m.avgStakeMultiplier:.3f}x', money(m.compoundPnl), money(m.maxDrawdown), money(m.returnDrawdown)])
    add_row('当前061固定1%', '180天', b180)
    add_row('当前061固定1%', '365天', b365)
    if ranked:
        best = ranked[0]
        add_row('V2综合最强', '180天', WindowMetrics(**best['metrics']['180d']))
        add_row('V2综合最强', '365天', WindowMetrics(**best['metrics']['365d']))
    if strict:
        bests = strict[0]
        add_row('V2严格通过最强', '180天', WindowMetrics(**bests['metrics']['180d']))
        add_row('V2严格通过最强', '365天', WindowMetrics(**bests['metrics']['365d']))
    md = ['# 061 贝叶斯动态仓位 V2 对比', '', '|配置|窗口|交易数|胜/负|胜率|平均仓位倍数|盈亏|最大回撤|收益回撤比|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        md.append('|' + '|'.join(row) + '|')
    md.append('')
    if off_base:
        md.append('## 官方实际订单回放')
        md.append('')
        md.append('|配置|订单数|胜/负|胜率|动态重算盈亏|原始真实金额盈亏|最大回撤|平均买价|')
        md.append('|---|---:|---:|---:|---:|---:|---:|---:|')
        md.append(f'|当前061固定1%|{off_base.trades}|{off_base.wins}/{off_base.losses}|{pct(off_base.winRate)}|{money(off_base.actualPriceDynamicPnl or off_base.compoundPnl)}|{money(off_base.originalLiveAmountPnl or 0.0)}|{money(off_base.maxDrawdown)}|{off_base.weightedAvgBuyPrice:.4f}|')
        if ranked and ranked[0]['metrics']['officialActualReplay']:
            mo = WindowMetrics(**ranked[0]['metrics']['officialActualReplay'])
            md.append(f'|V2综合最强|{mo.trades}|{mo.wins}/{mo.losses}|{pct(mo.winRate)}|{money(mo.actualPriceDynamicPnl or mo.compoundPnl)}|{money(mo.originalLiveAmountPnl or 0.0)}|{money(mo.maxDrawdown)}|{(mo.weightedAvgBuyPrice or 0):.4f}|')
        if strict and strict[0]['metrics']['officialActualReplay']:
            mo = WindowMetrics(**strict[0]['metrics']['officialActualReplay'])
            md.append(f'|V2严格通过最强|{mo.trades}|{mo.wins}/{mo.losses}|{pct(mo.winRate)}|{money(mo.actualPriceDynamicPnl or mo.compoundPnl)}|{money(mo.originalLiveAmountPnl or 0.0)}|{money(mo.maxDrawdown)}|{(mo.weightedAvgBuyPrice or 0):.4f}|')
        md.append('')
    md.append(f'严格通过候选数: {len(strict)}')
    md.append(f'结论: {unique["decision"]}')
    (REPORTS / 'top159_061_bayesian_sizer_v2_180_365_official_compare_latest.md').write_text('\n'.join(md), encoding='utf-8')
    write_json(REPORTS / 'top159_061_bayesian_sizer_v2_180_365_official_compare_latest.json', unique)

    print('\n'.join(md))


if __name__ == '__main__':
    run()
