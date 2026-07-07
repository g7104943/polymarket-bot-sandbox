#!/usr/bin/env python3
"""Research-only 061 V4: V2 profit engine + portfolio risk overlay.

Does not mutate live trading. It reuses the audited V2 data chain and applies a
portfolio-level overlay to V2 raw stake suggestions.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path('/Users/mac/polyfun')
SCRIPTS = ROOT / 'polyfun-next' / 'scripts'
REPORTS = ROOT / 'reports'
REPORTS.mkdir(parents=True, exist_ok=True)
V2_PATH = SCRIPTS / 'run_top159_061_bayesian_sizer_v2_search.py'
V3_PATH = SCRIPTS / 'run_top159_061_risk_budget_sizer_v3.py'


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


v2 = load_module(V2_PATH, 'v4_v2base')
v3 = load_module(V3_PATH, 'v4_v3base')

BUY_PRICE = v2.BUY_PRICE
INITIAL_FUNDS = v2.INITIAL_FUNDS
STAKE_BASE = v2.STAKE_BASE
WindowMetrics = v2.WindowMetrics

OUT_AUDIT_JSON = REPORTS / 'top159_061_v2_risk_overlay_v4_bug_audit_latest.json'
OUT_AUDIT_MD = REPORTS / 'top159_061_v2_risk_overlay_v4_bug_audit_latest.md'
OUT_LEADERBOARD_JSON = REPORTS / 'top159_061_v2_risk_overlay_v4_leaderboard_latest.json'
OUT_LEADERBOARD_MD = REPORTS / 'top159_061_v2_risk_overlay_v4_leaderboard_latest.md'
OUT_COMPARE_JSON = REPORTS / 'top159_061_v2_risk_overlay_v4_180_365_official_compare_latest.json'
OUT_COMPARE_MD = REPORTS / 'top159_061_v2_risk_overlay_v4_180_365_official_compare_latest.md'
OUT_VERDICT_JSON = REPORTS / 'top159_061_v2_risk_overlay_v4_unique_verdict_latest.json'
OUT_VERDICT_MD = REPORTS / 'top159_061_v2_risk_overlay_v4_unique_verdict_latest.md'

# V2 high-return engine from latest accepted V2 report.
V2_ENGINE_POLICY = {
    'name': 'v2_28a74d45d7ddbbe2',
    'dims': ['score_bin', 'direction', '1h_trend_bucket', '4h_pos_bucket'],
    'prior_strength': 400.0,
    'min_count': 120.0,
    'lookback_days': 60,
    'kelly_fraction': 0.35,
    'min_stake': 0.0035,
    'max_stake': 0.018,
    'edge_buffer': 0.0,
    'allow_skip': False,
    'skip_prob': 0.0,
    'weak_window': 0,
    'weak_win_min': 0.0,
    'weak_cap': 0.018,
    'weak_dd': 1.0,
    'update_lag': 2,
    'use_actual_price': False,
}


@dataclass(frozen=True)
class OverlayPolicy:
    name: str
    dd_soft: float
    dd_hard: float
    profit_lock: float
    soft_mult: float
    hard_mult: float
    lock_mult: float
    weak_window: int
    weak_win_min: float
    weak_mult: float
    loss_streak_trigger: int
    loss_streak_mult: float
    slope_window: int
    slope_mult: float
    recovery_level: float
    max_stake_cap: float
    min_stake_cap: float
    use_profit_lock: bool = True


def digest(parts: Iterable[Any]) -> str:
    return v2.digest_rows(parts)


def pct(x: float) -> str:
    return f'{x * 100:.2f}%'


def money(x: float) -> str:
    return f'{x:,.2f}'


def raw_v2_policy(use_actual_price: bool = False) -> v2.V2Policy:
    d = dict(V2_ENGINE_POLICY)
    d['use_actual_price'] = use_actual_price
    d['dims'] = tuple(d['dims'])
    return v2.V2Policy(**d)


def raw_v2_stake_from_counter(row: pd.Series, counter: Any, prior: float, policy: v2.V2Policy, q: float) -> Tuple[float, float, Tuple[str, ...]]:
    now_ns = v2.row_ts_ns(row)
    p, eff_n, used_level = counter.estimate(row, now_ns, prior)
    stake_pct, _ = v2.stake_from_p(policy, p, q, False)
    return stake_pct, p, used_level


def overlay_grid() -> List[OverlayPolicy]:
    out: List[OverlayPolicy] = []
    for dd_soft in [0.03, 0.05, 0.07]:
        for dd_hard in [0.08, 0.10, 0.12]:
            if dd_hard <= dd_soft:
                continue
            for profit_lock in [0.02, 0.03, 0.05]:
                for soft_mult, hard_mult, lock_mult in [(0.75, 0.45, 0.55), (0.65, 0.35, 0.50), (0.85, 0.55, 0.70)]:
                    for weak_window, weak_min in [(20, 0.50), (50, 0.535), (100, 0.545)]:
                        for recovery in [0.98, 0.99, 1.0]:
                            for loss_trigger in [3, 4, 5]:
                                nm = digest([dd_soft, dd_hard, profit_lock, soft_mult, hard_mult, lock_mult, weak_window, weak_min, recovery, loss_trigger])
                                out.append(OverlayPolicy(
                                    name=f'v4_{nm}', dd_soft=dd_soft, dd_hard=dd_hard, profit_lock=profit_lock,
                                    soft_mult=soft_mult, hard_mult=hard_mult, lock_mult=lock_mult,
                                    weak_window=weak_window, weak_win_min=weak_min, weak_mult=0.65,
                                    loss_streak_trigger=loss_trigger, loss_streak_mult=0.55,
                                    slope_window=weak_window, slope_mult=0.75,
                                    recovery_level=recovery, max_stake_cap=0.018, min_stake_cap=0.0035,
                                ))
    return out


def apply_overlay(policy: OverlayPolicy, raw_stake: float, funds: float, peak: float, last_lock_peak: float, recent: deque, equity_hist: deque, loss_streak: int) -> Tuple[float, str, float]:
    dd = (peak - funds) / peak if peak > 0 else 0.0
    mult = 1.0
    reason = 'raw'
    # Profit lock watches drawdown from the latest materially new high. It does
    # not use future information; it is updated online after every trade.
    if policy.use_profit_lock and last_lock_peak > 0 and (last_lock_peak - funds) / last_lock_peak >= policy.profit_lock:
        mult = min(mult, policy.lock_mult)
        reason = 'profit_lock'
    if dd >= policy.dd_hard:
        mult = min(mult, policy.hard_mult)
        reason = 'dd_hard'
    elif dd >= policy.dd_soft:
        mult = min(mult, policy.soft_mult)
        reason = 'dd_soft'
    if policy.weak_window and len(recent) >= policy.weak_window and float(np.mean(recent)) < policy.weak_win_min:
        mult = min(mult, policy.weak_mult)
        reason = 'weak_window'
    if loss_streak >= policy.loss_streak_trigger:
        mult = min(mult, policy.loss_streak_mult)
        reason = 'loss_streak'
    if len(equity_hist) >= policy.slope_window:
        # Negative equity slope over the recent completed trades triggers a mild cap.
        if float(equity_hist[-1]) < float(equity_hist[0]):
            mult = min(mult, policy.slope_mult)
            reason = 'negative_slope'
    if dd < policy.dd_soft and funds >= peak * policy.recovery_level:
        # Recovery state: allow raw stake again.
        if reason in {'dd_soft', 'profit_lock', 'negative_slope'}:
            mult = max(mult, 0.90)
            reason = 'recovered'
    stake = raw_stake * mult
    stake = min(max(stake, policy.min_stake_cap), policy.max_stake_cap)
    return stake, reason, last_lock_peak


def simulate_v4(train: pd.DataFrame, val: pd.DataFrame, overlay: OverlayPolicy, *, buy_price: float = BUY_PRICE, actual_mode: bool = False) -> WindowMetrics:
    raw_policy = raw_v2_policy(use_actual_price=actual_mode)
    dims = tuple(raw_policy.dims)
    needed = sorted(set(dims + ('dt', 'won', 'direction', 'score_bin')))
    train = v2._ensure_cols(train, needed).sort_values('dt').reset_index(drop=True)
    val = v2._ensure_cols(val, needed).sort_values('dt').reset_index(drop=True)
    levels = v2.hierarchy_levels(dims)
    prior = v2.global_win_rate(train)
    counter = v2.HierarchicalCounter(levels, raw_policy.prior_strength, raw_policy.min_count, raw_policy.lookback_days)
    for _, row in train.iterrows():
        counter.add(row, v2.row_ts_ns(row), bool(row['won']))

    funds = INITIAL_FUNDS
    peak = funds
    last_lock_peak = funds
    max_dd = 0.0
    trades = wins = losses = skipped = 0
    down = up = strong = weak_n = 0
    stake_pcts: List[float] = []
    hash_parts: List[Any] = []
    pending: deque = deque()
    recent: deque = deque(maxlen=max(overlay.weak_window, 1))
    equity_hist: deque = deque(maxlen=max(overlay.slope_window, 1))
    loss_streak = 0
    original_live_amount_pnl = 0.0 if actual_mode else None
    weighted_cost = 0.0
    weighted_amt = 0.0

    def flush_until(idx: int):
        while pending and pending[0][0] <= idx:
            _, prow = pending.popleft()
            counter.add(prow, v2.row_ts_ns(prow), bool(prow['won']))
            recent.append(1 if bool(prow['won']) else 0)

    for i, row in val.iterrows():
        flush_until(i)
        q = buy_price
        if actual_mode and 'actual_avg_price' in row and pd.notna(row['actual_avg_price']):
            q = float(row['actual_avg_price'])
            if not (0.01 <= q <= 0.99):
                q = buy_price
        raw_stake, p, used_level = raw_v2_stake_from_counter(row, counter, prior, raw_policy, q)
        stake_pct, reason, last_lock_peak = apply_overlay(overlay, raw_stake, funds, peak, last_lock_peak, recent, equity_hist, loss_streak)
        pending.append((i + raw_policy.update_lag, row))

        stake = funds * stake_pct
        won = bool(row['won'])
        if won:
            pnl = stake * (1.0 / q - 1.0)
            wins += 1
            loss_streak = 0
        else:
            pnl = -stake
            losses += 1
            loss_streak += 1
        funds += pnl
        peak = max(peak, funds)
        if peak > last_lock_peak:
            last_lock_peak = peak
        max_dd = max(max_dd, peak - funds)
        equity_hist.append(funds)
        trades += 1
        stake_pcts.append(stake_pct)
        if stake_pct < STAKE_BASE * 0.9:
            down += 1
        elif stake_pct > STAKE_BASE * 1.4:
            strong += 1
        elif stake_pct > STAKE_BASE * 1.05:
            up += 1
        if reason in {'weak_window', 'loss_streak', 'negative_slope', 'dd_soft', 'dd_hard', 'profit_lock'}:
            weak_n += 1
        if actual_mode:
            if original_live_amount_pnl is not None:
                original_live_amount_pnl += float(row.get('actual_pnl', 0.0) or 0.0)
            amt = float(row.get('actual_cost', 0.0) or 0.0)
            if amt > 0 and 'actual_avg_price' in row and pd.notna(row['actual_avg_price']):
                weighted_cost += amt * float(row['actual_avg_price'])
                weighted_amt += amt
        hash_parts.append((row.get('market_slug', row.get('market', i)), reason, round(p, 6), round(raw_stake, 6), round(stake_pct, 6), won, round(q, 5), tuple(used_level)))

    pnl = funds - INITIAL_FUNDS
    avg = float(np.mean(stake_pcts)) if stake_pcts else 0.0
    return WindowMetrics(
        rows=int(len(val)), trades=int(trades), skipped=int(skipped), wins=int(wins), losses=int(losses),
        winRate=float(wins / trades if trades else 0.0), avgStakePct=avg, avgStakeMultiplier=avg / STAKE_BASE if STAKE_BASE else 0.0,
        compoundPnl=float(pnl), endingFunds=float(funds), maxDrawdown=float(max_dd), returnDrawdown=float(pnl / max_dd) if max_dd else 0.0,
        downsizeCount=int(down), upsizeCount=int(up), strongUpsizeCount=int(strong), weakRegimeCount=int(weak_n), setHash=v2.digest_rows(hash_parts),
        originalLiveAmountPnl=float(original_live_amount_pnl) if original_live_amount_pnl is not None else None,
        actualPriceDynamicPnl=float(pnl) if actual_mode else None,
        weightedAvgBuyPrice=float(weighted_cost / weighted_amt) if weighted_amt > 0 else None,
    )


def precompute_raw_v2_inputs(train: pd.DataFrame, val: pd.DataFrame, *, buy_price: float = BUY_PRICE, actual_mode: bool = False) -> pd.DataFrame:
    """Precompute the V2 engine output once; V4 overlays only scale that output.

    This is equivalent to the first half of simulate_v4().  The expensive
    hierarchical counter is independent from the overlay parameters, so reusing
    it avoids rebuilding the same counter for every V4 candidate.
    """
    raw_policy = raw_v2_policy(use_actual_price=actual_mode)
    dims = tuple(raw_policy.dims)
    needed = sorted(set(dims + ('dt', 'won', 'direction', 'score_bin')))
    train = v2._ensure_cols(train, needed).sort_values('dt').reset_index(drop=True)
    val = v2._ensure_cols(val, needed).sort_values('dt').reset_index(drop=True).copy()
    levels = v2.hierarchy_levels(dims)
    prior = v2.global_win_rate(train)
    counter = v2.HierarchicalCounter(levels, raw_policy.prior_strength, raw_policy.min_count, raw_policy.lookback_days)
    for _, row in train.iterrows():
        counter.add(row, v2.row_ts_ns(row), bool(row['won']))

    pending: deque = deque()
    raw_stakes: List[float] = []
    probs: List[float] = []
    used_levels: List[Tuple[str, ...]] = []
    qs: List[float] = []

    def flush_until(idx: int):
        while pending and pending[0][0] <= idx:
            _, prow = pending.popleft()
            counter.add(prow, v2.row_ts_ns(prow), bool(prow['won']))

    for i, row in val.iterrows():
        flush_until(i)
        q = buy_price
        if actual_mode and 'actual_avg_price' in row and pd.notna(row['actual_avg_price']):
            q = float(row['actual_avg_price'])
            if not (0.01 <= q <= 0.99):
                q = buy_price
        raw_stake, p, used_level = raw_v2_stake_from_counter(row, counter, prior, raw_policy, q)
        pending.append((i + raw_policy.update_lag, row))
        raw_stakes.append(float(raw_stake))
        probs.append(float(p))
        used_levels.append(tuple(used_level))
        qs.append(float(q))

    val['_v4_raw_stake'] = raw_stakes
    val['_v4_p'] = probs
    val['_v4_used_level'] = used_levels
    val['_v4_q'] = qs
    val['_v4_update_lag'] = int(raw_policy.update_lag)
    return val


def simulate_v4_precomputed(val: pd.DataFrame, overlay: OverlayPolicy, *, actual_mode: bool = False) -> WindowMetrics:
    funds = INITIAL_FUNDS
    peak = funds
    last_lock_peak = funds
    max_dd = 0.0
    trades = wins = losses = skipped = 0
    down = up = strong = weak_n = 0
    stake_pcts: List[float] = []
    hash_parts: List[Any] = []
    update_lag = int(val['_v4_update_lag'].iloc[0]) if len(val) else 2
    pending: deque = deque()
    recent: deque = deque(maxlen=max(overlay.weak_window, 1))
    equity_hist: deque = deque(maxlen=max(overlay.slope_window, 1))
    loss_streak = 0
    original_live_amount_pnl = 0.0 if actual_mode else None
    weighted_cost = 0.0
    weighted_amt = 0.0

    def flush_until(idx: int):
        while pending and pending[0][0] <= idx:
            _, won_done = pending.popleft()
            recent.append(1 if bool(won_done) else 0)

    for i, row in val.iterrows():
        flush_until(i)
        q = float(row['_v4_q'])
        raw_stake = float(row['_v4_raw_stake'])
        p = float(row['_v4_p'])
        used_level = tuple(row['_v4_used_level'])
        stake_pct, reason, last_lock_peak = apply_overlay(overlay, raw_stake, funds, peak, last_lock_peak, recent, equity_hist, loss_streak)
        won = bool(row['won'])
        pending.append((i + update_lag, won))

        stake = funds * stake_pct
        if won:
            pnl = stake * (1.0 / q - 1.0)
            wins += 1
            loss_streak = 0
        else:
            pnl = -stake
            losses += 1
            loss_streak += 1
        funds += pnl
        peak = max(peak, funds)
        if peak > last_lock_peak:
            last_lock_peak = peak
        max_dd = max(max_dd, peak - funds)
        equity_hist.append(funds)
        trades += 1
        stake_pcts.append(stake_pct)
        if stake_pct < STAKE_BASE * 0.9:
            down += 1
        elif stake_pct > STAKE_BASE * 1.4:
            strong += 1
        elif stake_pct > STAKE_BASE * 1.05:
            up += 1
        if reason in {'weak_window', 'loss_streak', 'negative_slope', 'dd_soft', 'dd_hard', 'profit_lock'}:
            weak_n += 1
        if actual_mode:
            if original_live_amount_pnl is not None:
                original_live_amount_pnl += float(row.get('actual_pnl', 0.0) or 0.0)
            amt = float(row.get('actual_cost', 0.0) or 0.0)
            if amt > 0 and pd.notna(row.get('actual_avg_price', np.nan)):
                weighted_cost += amt * float(row['actual_avg_price'])
                weighted_amt += amt
        hash_parts.append((row.get('market_slug', row.get('market', i)), reason, round(p, 6), round(raw_stake, 6), round(stake_pct, 6), won, round(q, 5), used_level))

    pnl = funds - INITIAL_FUNDS
    avg = float(np.mean(stake_pcts)) if stake_pcts else 0.0
    return WindowMetrics(
        rows=int(len(val)), trades=int(trades), skipped=int(skipped), wins=int(wins), losses=int(losses),
        winRate=float(wins / trades if trades else 0.0), avgStakePct=avg, avgStakeMultiplier=avg / STAKE_BASE if STAKE_BASE else 0.0,
        compoundPnl=float(pnl), endingFunds=float(funds), maxDrawdown=float(max_dd), returnDrawdown=float(pnl / max_dd) if max_dd else 0.0,
        downsizeCount=int(down), upsizeCount=int(up), strongUpsizeCount=int(strong), weakRegimeCount=int(weak_n), setHash=v2.digest_rows(hash_parts),
        originalLiveAmountPnl=float(original_live_amount_pnl) if original_live_amount_pnl is not None else None,
        actualPriceDynamicPnl=float(pnl) if actual_mode else None,
        weightedAvgBuyPrice=float(weighted_cost / weighted_amt) if weighted_amt > 0 else None,
    )


def metrics_to_dict(m: WindowMetrics) -> Dict[str, Any]:
    return v2.metrics_to_dict(m)


def pass_gate(m180: WindowMetrics, m365: WindowMetrics, off: Optional[WindowMetrics], b180: WindowMetrics, b365: WindowMetrics, boff: Optional[WindowMetrics]) -> Tuple[bool, List[str]]:
    reasons = []
    if m180.compoundPnl < b180.compoundPnl * 1.10:
        reasons.append('180盈亏未高于061 10%')
    if m365.compoundPnl < b365.compoundPnl * 1.20:
        reasons.append('365盈亏未高于061 20%')
    if m180.maxDrawdown > b180.maxDrawdown * 1.05:
        reasons.append('180回撤高于061超过5%')
    if m365.maxDrawdown > b365.maxDrawdown * 1.05:
        reasons.append('365回撤高于061超过5%')
    if m180.returnDrawdown < b180.returnDrawdown:
        reasons.append('180收益回撤比低于061')
    if m365.returnDrawdown < b365.returnDrawdown:
        reasons.append('365收益回撤比低于061')
    if m365.trades < b365.trades * 0.95:
        reasons.append('365交易数未保留95%')
    if off is not None and boff is not None:
        if (off.actualPriceDynamicPnl or off.compoundPnl) < (boff.actualPriceDynamicPnl or boff.compoundPnl):
            reasons.append('官方实际价格动态回放盈亏低于061')
        if off.maxDrawdown > boff.maxDrawdown * 1.10:
            reasons.append('官方实际价格动态回放回撤明显更差')
    return len(reasons) == 0, reasons


def random_label_audit(data: Dict[str, Any], overlays: List[OverlayPolicy], b180: WindowMetrics, b365: WindowMetrics) -> Dict[str, Any]:
    total = 0
    sampled = min(20, len(overlays))
    by_seed = []
    for seed in [20260513, 20260514, 20260515]:
        rng = np.random.default_rng(seed)
        tr180 = data['train180'].copy(); tr365 = data['train365'].copy(); val180 = data['val180'].copy(); val365 = data['val365'].copy()
        for df in [tr180, tr365, val180, val365]:
            df['won'] = rng.permutation(df['won'].to_numpy())
        pre180 = precompute_raw_v2_inputs(tr180, val180)
        pre365 = precompute_raw_v2_inputs(tr365, val365)
        strict = 0
        best = -1e18
        for ov in overlays[:sampled]:
            m180 = simulate_v4_precomputed(pre180, ov)
            m365 = simulate_v4_precomputed(pre365, ov)
            if m180.compoundPnl >= b180.compoundPnl * 1.10 and m365.compoundPnl >= b365.compoundPnl * 1.20 and m180.maxDrawdown <= b180.maxDrawdown * 1.05 and m365.maxDrawdown <= b365.maxDrawdown * 1.05:
                strict += 1
            best = max(best, m365.compoundPnl)
        total += strict
        by_seed.append({'seed': seed, 'strictPassCount': strict, 'bestRandom365Pnl': best})
    return {'passed': total == 0, 'strictPassCount': total, 'sampledPolicies': sampled, 'bySeed': by_seed}


def feature_shift_audit(data: Dict[str, Any], overlay: OverlayPolicy) -> Dict[str, Any]:
    val = data['val365'].copy()
    for c in V2_ENGINE_POLICY['dims']:
        if c in val.columns:
            val[c] = val[c].shift(1).fillna(val[c].iloc[0])
    pre_normal = precompute_raw_v2_inputs(data['train365'], data['val365'])
    pre_shifted = precompute_raw_v2_inputs(data['train365'], val)
    normal = simulate_v4_precomputed(pre_normal, overlay)
    shifted = simulate_v4_precomputed(pre_shifted, overlay)
    return {'passed': normal.setHash != shifted.setHash or abs(normal.compoundPnl - shifted.compoundPnl) > 1e-9, 'normalHash': normal.setHash, 'shiftedHash': shifted.setHash, 'normal365Pnl': normal.compoundPnl, 'shifted365Pnl': shifted.compoundPnl}


def run() -> None:
    data = v2.prepare_data()
    b180 = v2.fixed_metrics(data['val180'])
    b365 = v2.fixed_metrics(data['val365'])
    off_base = v2.fixed_metrics(data['official_rows'], actual_mode=True) if not data['official_rows'].empty else None
    v2_pol = raw_v2_policy()
    v2_180 = v2.simulate_v2(data['train180'], data['val180'], v2_pol)
    v2_365 = v2.simulate_v2(data['train365'], data['val365'], v2_pol)
    v3_verdict = None
    try:
        v3_verdict = json.loads((REPORTS / 'top159_061_risk_budget_sizer_v3_unique_verdict_latest.json').read_text())
    except Exception:
        v3_verdict = None
    overlays = overlay_grid()
    baseline_ok = (b180.trades == v2.BASELINE_061['180d']['trades'] and round(b180.compoundPnl, 6) == round(v2.BASELINE_061['180d']['compoundPnl'], 6) and b365.trades == v2.BASELINE_061['365d']['trades'] and round(b365.compoundPnl, 6) == round(v2.BASELINE_061['365d']['compoundPnl'], 6))
    audit = {'baselineReproductionPassed': baseline_ok, 'baseline180': metrics_to_dict(b180), 'baseline365': metrics_to_dict(b365), 'v2Reproduction': {'180d': metrics_to_dict(v2_180), '365d': metrics_to_dict(v2_365)}, 'officialBaseline': metrics_to_dict(off_base) if off_base else None, 'policyCount': len(overlays)}
    if baseline_ok:
        audit['randomLabel'] = random_label_audit(data, overlays, b180, b365)
        audit['featureShift'] = feature_shift_audit(data, overlays[0])
        audit['passed'] = bool(audit['randomLabel']['passed'] and audit['featureShift']['passed'])
    else:
        audit['passed'] = False
    v2.write_json(OUT_AUDIT_JSON, audit)
    OUT_AUDIT_MD.write_text('# 061 V4 审计\n\n' + f'- 061复现: {"通过" if baseline_ok else "失败"}\n- V2复现365盈亏: `{v2_365.compoundPnl:,.2f}` 回撤 `{v2_365.maxDrawdown:,.2f}`\n- 随机标签: {"通过" if audit.get("randomLabel",{}).get("passed") else "失败"}\n- 特征错位: {"通过" if audit.get("featureShift",{}).get("passed") else "失败"}\n- 候选数: `{len(overlays)}`\n', encoding='utf-8')
    if not audit['passed']:
        raise SystemExit('audit failed')

    pre180 = precompute_raw_v2_inputs(data['train180'], data['val180'])
    pre365 = precompute_raw_v2_inputs(data['train365'], data['val365'])
    preoff = precompute_raw_v2_inputs(data['official_train'], data['official_rows'], actual_mode=True) if off_base is not None and not data['official_rows'].empty else None

    rows = []
    for idx, ov in enumerate(overlays, 1):
        m180 = simulate_v4_precomputed(pre180, ov)
        m365 = simulate_v4_precomputed(pre365, ov)
        score = ((m180.compoundPnl - b180.compoundPnl) / abs(b180.compoundPnl) * 0.25 + (m365.compoundPnl - b365.compoundPnl) / abs(b365.compoundPnl) * 0.40 + (b365.maxDrawdown - m365.maxDrawdown) / b365.maxDrawdown * 0.25 + (m365.returnDrawdown - b365.returnDrawdown) / max(abs(b365.returnDrawdown), 1) * 0.10)
        rows.append({'rankScore': score, 'candidateId': v2.digest_rows([ov.name, m180.setHash, m365.setHash]), 'policy': asdict(ov), 'strictPass': False, 'failReasons': [], 'metrics': {'180d': metrics_to_dict(m180), '365d': metrics_to_dict(m365), 'officialActualReplay': None}})
        if idx % 500 == 0:
            v2.write_json(OUT_LEADERBOARD_JSON, sorted(rows, key=lambda r: r['rankScore'], reverse=True)[:300])
    ranked = sorted(rows, key=lambda r: r['rankScore'], reverse=True)
    if off_base is not None and not data['official_rows'].empty:
        for r in ranked[:120]:
            ov = OverlayPolicy(**r['policy'])
            off = simulate_v4_precomputed(preoff, ov, actual_mode=True)
            r['metrics']['officialActualReplay'] = metrics_to_dict(off)
            passed, reasons = pass_gate(WindowMetrics(**r['metrics']['180d']), WindowMetrics(**r['metrics']['365d']), off, b180, b365, off_base)
            r['strictPass'] = passed
            r['failReasons'] = reasons
            r['rankScore'] += ((off.actualPriceDynamicPnl - off_base.actualPriceDynamicPnl) / max(abs(off_base.actualPriceDynamicPnl), 1) * 0.10 if off.actualPriceDynamicPnl is not None and off_base.actualPriceDynamicPnl is not None else 0.0)
        ranked = sorted(rows, key=lambda r: r['rankScore'], reverse=True)
    repeat = []
    for r in ranked[:10]:
        ov = OverlayPolicy(**r['policy'])
        a180 = simulate_v4_precomputed(pre180, ov)
        a365 = simulate_v4_precomputed(pre365, ov)
        repeat.append({'candidateId': r['candidateId'], 'passed': a180.setHash == r['metrics']['180d']['setHash'] and a365.setHash == r['metrics']['365d']['setHash'], 'repeat180Hash': a180.setHash, 'repeat365Hash': a365.setHash})
    strict = [r for r in ranked if r['strictPass']]
    verdict = {'baseline': {'180d': metrics_to_dict(b180), '365d': metrics_to_dict(b365), 'officialActualReplay': metrics_to_dict(off_base) if off_base else None}, 'v2Engine': {'180d': metrics_to_dict(v2_180), '365d': metrics_to_dict(v2_365)}, 'v3PriorBest': v3_verdict.get('bestOverall') if v3_verdict else None, 'bestOverall': ranked[0] if ranked else None, 'bestStrictPass': strict[0] if strict else None, 'strictPassCount': len(strict), 'repeatChecks': repeat, 'auditPassed': audit['passed'], 'decision': 'FOUND_V4_CANDIDATE_FOR_SHADOW_ONLY' if strict else 'NO_V4_CANDIDATE; keep live 061'}
    v2.write_json(OUT_LEADERBOARD_JSON, ranked[:500])
    v2.write_json(OUT_VERDICT_JSON, verdict)

    def row(name: str, window: str, m: WindowMetrics) -> str:
        return f'|{name}|{window}|{m.trades}|{m.wins}/{m.losses}|{m.winRate*100:.2f}%|{m.avgStakeMultiplier:.3f}x|{m.compoundPnl:,.2f}|{m.maxDrawdown:,.2f}|{m.returnDrawdown:.2f}|'
    lines = ['# 061 V4：V2收益引擎 + 回撤预算对比', '', '|配置|窗口|交易数|胜/负|胜率|平均仓位倍数|盈亏|最大回撤|收益回撤比|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    lines += [row('当前061固定1%', '180天', b180), row('当前061固定1%', '365天', b365), row('V2收益引擎', '180天', v2_180), row('V2收益引擎', '365天', v2_365)]
    if ranked:
        lines += [row('V4综合最强', '180天', WindowMetrics(**ranked[0]['metrics']['180d'])), row('V4综合最强', '365天', WindowMetrics(**ranked[0]['metrics']['365d']))]
    if strict:
        lines += [row('V4严格通过最强', '180天', WindowMetrics(**strict[0]['metrics']['180d'])), row('V4严格通过最强', '365天', WindowMetrics(**strict[0]['metrics']['365d']))]
    lines += ['', '## 官方实际订单回放', '', '|配置|订单数|胜/负|胜率|动态重算盈亏|同批订单原始金额盈亏(非策略结果)|最大回撤|平均买价|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    if off_base:
        lines.append(f'|当前061固定1%|{off_base.trades}|{off_base.wins}/{off_base.losses}|{off_base.winRate*100:.2f}%|{(off_base.actualPriceDynamicPnl or off_base.compoundPnl):,.2f}|{(off_base.originalLiveAmountPnl or 0.0):,.2f}|{off_base.maxDrawdown:,.2f}|{(off_base.weightedAvgBuyPrice or 0):.4f}|')
    if ranked and ranked[0]['metrics'].get('officialActualReplay'):
        mo = WindowMetrics(**ranked[0]['metrics']['officialActualReplay'])
        lines.append(f'|V4综合最强|{mo.trades}|{mo.wins}/{mo.losses}|{mo.winRate*100:.2f}%|{(mo.actualPriceDynamicPnl or mo.compoundPnl):,.2f}|{(mo.originalLiveAmountPnl or 0.0):,.2f}|{mo.maxDrawdown:,.2f}|{(mo.weightedAvgBuyPrice or 0):.4f}|')
    if strict and strict[0]['metrics'].get('officialActualReplay'):
        mo = WindowMetrics(**strict[0]['metrics']['officialActualReplay'])
        lines.append(f'|V4严格通过最强|{mo.trades}|{mo.wins}/{mo.losses}|{mo.winRate*100:.2f}%|{(mo.actualPriceDynamicPnl or mo.compoundPnl):,.2f}|{(mo.originalLiveAmountPnl or 0.0):,.2f}|{mo.maxDrawdown:,.2f}|{(mo.weightedAvgBuyPrice or 0):.4f}|')
    lines += ['', f'严格通过候选数: {len(strict)}', f'结论: {verdict["decision"]}']
    OUT_COMPARE_MD.write_text('\n'.join(lines), encoding='utf-8')
    v2.write_json(OUT_COMPARE_JSON, verdict)
    OUT_LEADERBOARD_MD.write_text('# 061 V4 排行榜\n\n' + '\n'.join([f"{i+1}. `{r['candidateId']}` score={r['rankScore']:.4f} pass={r['strictPass']} 365pnl={r['metrics']['365d']['compoundPnl']:.2f} dd={r['metrics']['365d']['maxDrawdown']:.2f}" for i, r in enumerate(ranked[:100])]), encoding='utf-8')
    OUT_VERDICT_MD.write_text('# 061 V4 唯一结论\n\n' + f'- 严格通过候选数: `{len(strict)}`\n- 结论: `{verdict["decision"]}`\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    run()
