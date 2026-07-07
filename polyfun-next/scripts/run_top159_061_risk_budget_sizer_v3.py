#!/usr/bin/env python3
"""Research-only 061 risk-budget dynamic sizer V3.

Does not mutate live trading. Reuses the audited 061/V2 data chain, then tests
risk-budget sizing intended to keep V2-like upside while capping drawdown.
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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


v2 = load_module(V2_PATH, 'risk_v3_v2base')

BUY_PRICE = v2.BUY_PRICE
INITIAL_FUNDS = v2.INITIAL_FUNDS
STAKE_BASE = v2.STAKE_BASE
WindowMetrics = v2.WindowMetrics

OUT_AUDIT_JSON = REPORTS / 'top159_061_risk_budget_sizer_v3_bug_audit_latest.json'
OUT_AUDIT_MD = REPORTS / 'top159_061_risk_budget_sizer_v3_bug_audit_latest.md'
OUT_LEADERBOARD_JSON = REPORTS / 'top159_061_risk_budget_sizer_v3_leaderboard_latest.json'
OUT_LEADERBOARD_MD = REPORTS / 'top159_061_risk_budget_sizer_v3_leaderboard_latest.md'
OUT_COMPARE_JSON = REPORTS / 'top159_061_risk_budget_sizer_v3_180_365_official_compare_latest.json'
OUT_COMPARE_MD = REPORTS / 'top159_061_risk_budget_sizer_v3_180_365_official_compare_latest.md'
OUT_VERDICT_JSON = REPORTS / 'top159_061_risk_budget_sizer_v3_unique_verdict_latest.json'
OUT_VERDICT_MD = REPORTS / 'top159_061_risk_budget_sizer_v3_unique_verdict_latest.md'


@dataclass(frozen=True)
class V3Policy:
    name: str
    style: str
    dims: Tuple[str, ...]
    prior_strength: float
    min_count: float
    lookback_days: Optional[int]
    kelly_fraction: float
    min_stake: float
    max_stake: float
    observe_max_stake: float
    dd_soft: float
    dd_mid: float
    dd_hard: float
    dd_soft_cap: float
    dd_mid_cap: float
    dd_hard_cap: float
    weak_window: int
    weak_win_min: float
    weak_cap: float
    vol_cap: float
    high_p: float
    risk_on_p: float
    update_lag: int
    use_actual_price: bool = False


def digest(parts: Iterable[Any]) -> str:
    return v2.digest_rows(parts)


def pct(x: float) -> str:
    return f'{x * 100:.2f}%'


def money(x: float) -> str:
    return f'{x:,.2f}'


def row_vol_high(row: pd.Series) -> bool:
    return str(row.get('1h_vol_bucket', '')) == 'high' or str(row.get('4h_vol_bucket', '')) == 'high' or str(row.get('1h_range_bucket', '')) == 'high' or str(row.get('4h_range_bucket', '')) == 'high'


def base_kelly_stake(p: float, buy_price: float, kelly_fraction: float, min_stake: float, max_stake: float) -> float:
    b = 1.0 / buy_price - 1.0
    if b <= 0:
        return STAKE_BASE
    kelly = (b * p - (1.0 - p)) / b
    raw = kelly_fraction * kelly
    if raw <= 0:
        raw = min_stake
    return min(max(raw, min_stake), max_stake)


def apply_risk_budget(policy: V3Policy, stake: float, p: float, row: pd.Series, funds: float, peak: float, recent: deque) -> Tuple[float, str, bool]:
    dd_pct = (peak - funds) / peak if peak > 0 else 0.0
    weak = False
    if policy.weak_window and len(recent) >= policy.weak_window and float(np.mean(recent)) < policy.weak_win_min:
        weak = True
    action = 'base'

    # Drawdown budget: as soon as equity is no longer near high water mark,
    # max allowed stake compresses before the drawdown gets large.
    if dd_pct >= policy.dd_hard:
        stake = min(stake, policy.dd_hard_cap)
        action = 'dd_hard_cap'
    elif dd_pct >= policy.dd_mid:
        stake = min(stake, policy.dd_mid_cap)
        action = 'dd_mid_cap'
    elif dd_pct >= policy.dd_soft:
        stake = min(stake, policy.dd_soft_cap)
        action = 'dd_soft_cap'

    if weak:
        stake = min(stake, policy.weak_cap)
        action = 'weak_cap'

    # High vol is where V2 tended to make large drawdowns. Only allow above-base
    # exposure in high vol if the calibrated probability is very strong.
    if row_vol_high(row) and p < policy.high_p:
        stake = min(stake, policy.vol_cap)
        action = 'vol_cap' if action == 'base' else action + '+vol'

    # Observation max is allowed only in clean risk-on state.
    risk_on = (dd_pct < policy.dd_soft and not weak and (not row_vol_high(row) or p >= policy.high_p) and p >= policy.risk_on_p)
    if not risk_on:
        stake = min(stake, policy.max_stake)
    else:
        stake = min(stake, policy.observe_max_stake)

    if stake < STAKE_BASE * 0.9:
        if action == 'base': action = 'downsize'
    elif stake > STAKE_BASE * 1.40:
        action = 'strong_upsize'
    elif stake > STAKE_BASE * 1.05:
        action = 'upsize'
    return stake, action, weak


def simulate_v3(train: pd.DataFrame, val: pd.DataFrame, policy: V3Policy, *, buy_price: float = BUY_PRICE, actual_mode: bool = False) -> WindowMetrics:
    dims = tuple(policy.dims)
    needed = sorted(set(dims + ('dt', 'won', 'direction', 'score_bin', '1h_vol_bucket', '4h_vol_bucket', '1h_range_bucket', '4h_range_bucket')))
    train = v2._ensure_cols(train, needed).sort_values('dt').reset_index(drop=True)
    val = v2._ensure_cols(val, needed).sort_values('dt').reset_index(drop=True)
    levels = v2.hierarchy_levels(dims)
    prior = v2.global_win_rate(train)
    counter = v2.HierarchicalCounter(levels, policy.prior_strength, policy.min_count, policy.lookback_days)
    for _, row in train.iterrows():
        counter.add(row, v2.row_ts_ns(row), bool(row['won']))

    funds = INITIAL_FUNDS
    peak = funds
    max_dd = 0.0
    trades = wins = losses = skipped = 0
    down = up = strong = weak_n = 0
    stake_pcts: List[float] = []
    hash_parts: List[Any] = []
    pending: deque = deque()
    recent: deque = deque(maxlen=max(policy.weak_window, 1) if policy.weak_window else 1)
    original_live_amount_pnl = 0.0 if actual_mode else None
    weighted_cost = 0.0
    weighted_amt = 0.0

    def flush_until(idx: int):
        while pending and pending[0][0] <= idx:
            _, prow = pending.popleft()
            counter.add(prow, v2.row_ts_ns(prow), bool(prow['won']))
            if policy.weak_window:
                recent.append(1 if bool(prow['won']) else 0)

    for i, row in val.iterrows():
        flush_until(i)
        now_ns = v2.row_ts_ns(row)
        p, eff_n, used_level = counter.estimate(row, now_ns, prior)
        q = buy_price
        if actual_mode and policy.use_actual_price and 'actual_avg_price' in row and pd.notna(row['actual_avg_price']):
            q = float(row['actual_avg_price'])
            if not (0.01 <= q <= 0.99):
                q = buy_price
        stake_pct = base_kelly_stake(p, q, policy.kelly_fraction, policy.min_stake, policy.observe_max_stake)
        stake_pct, action, weak = apply_risk_budget(policy, stake_pct, p, row, funds, peak, recent)
        pending.append((i + policy.update_lag, row))

        stake = funds * stake_pct
        won = bool(row['won'])
        if won:
            pnl = stake * (1.0 / q - 1.0)
            wins += 1
        else:
            pnl = -stake
            losses += 1
        funds += pnl
        peak = max(peak, funds)
        max_dd = max(max_dd, peak - funds)
        trades += 1
        stake_pcts.append(stake_pct)
        if stake_pct < STAKE_BASE * 0.9:
            down += 1
        elif stake_pct > STAKE_BASE * 1.4:
            strong += 1
        elif stake_pct > STAKE_BASE * 1.05:
            up += 1
        if weak:
            weak_n += 1
        if actual_mode:
            if original_live_amount_pnl is not None:
                original_live_amount_pnl += float(row.get('actual_pnl', 0.0) or 0.0)
            amt = float(row.get('actual_cost', 0.0) or 0.0)
            if amt > 0 and 'actual_avg_price' in row and pd.notna(row['actual_avg_price']):
                weighted_cost += amt * float(row['actual_avg_price'])
                weighted_amt += amt
        hash_parts.append((row.get('market_slug', row.get('market', i)), action, round(p, 6), round(stake_pct, 6), won, round(q, 5), tuple(used_level)))

    pnl = funds - INITIAL_FUNDS
    avg = float(np.mean(stake_pcts)) if stake_pcts else 0.0
    return WindowMetrics(
        rows=int(len(val)), trades=int(trades), skipped=int(skipped), wins=int(wins), losses=int(losses),
        winRate=float(wins / trades if trades else 0.0), avgStakePct=avg, avgStakeMultiplier=avg / STAKE_BASE if STAKE_BASE else 0.0,
        compoundPnl=float(pnl), endingFunds=float(funds), maxDrawdown=float(max_dd), returnDrawdown=float(pnl / max_dd) if max_dd else 0.0,
        downsizeCount=int(down), upsizeCount=int(up), strongUpsizeCount=int(strong), weakRegimeCount=int(weak_n),
        setHash=digest(hash_parts), originalLiveAmountPnl=float(original_live_amount_pnl) if original_live_amount_pnl is not None else None,
        actualPriceDynamicPnl=float(pnl) if actual_mode else None, weightedAvgBuyPrice=float(weighted_cost / weighted_amt) if weighted_amt > 0 else None,
    )


def policy_grid() -> List[V3Policy]:
    dims_options = [
        ('score_bin', 'direction'),
        ('score_bin', 'direction', '1h_vol_bucket', '4h_vol_bucket'),
        ('score_bin', 'direction', '1h_trend_bucket', '4h_pos_bucket'),
        ('score_bin', 'direction', 'hour_bucket', '1h_vol_bucket'),
    ]
    styles = {
        'conservative': dict(max_stake=[0.0115, 0.0125], observe=[0.015], dd_soft=[0.025], dd_mid=[0.055], dd_hard=[0.085], caps=[(0.010, 0.0075, 0.005)]),
        'balanced': dict(max_stake=[0.0125, 0.015], observe=[0.018], dd_soft=[0.035], dd_mid=[0.070], dd_hard=[0.105], caps=[(0.0125, 0.009, 0.006)]),
        'aggressive': dict(max_stake=[0.015], observe=[0.018], dd_soft=[0.045], dd_mid=[0.085], dd_hard=[0.13], caps=[(0.0135, 0.010, 0.0075)]),
    }
    out: List[V3Policy] = []
    for style, cfg in styles.items():
        for dims in dims_options:
            for prior in [200.0, 400.0]:
                for min_count in [60.0, 120.0]:
                    for lookback in [None, 60]:
                        for kf in ([0.12, 0.20] if style == 'conservative' else [0.20, 0.30]):
                            for max_stake in cfg['max_stake']:
                                for observe in cfg['observe']:
                                    for caps in cfg['caps']:
                                        for weak_window in [20, 50]:
                                            nm = digest([style, dims, prior, min_count, lookback, kf, max_stake, observe, caps, weak_window])
                                            out.append(V3Policy(
                                                name=f'v3_{style}_{nm}', style=style, dims=tuple(dims), prior_strength=prior,
                                                min_count=min_count, lookback_days=lookback, kelly_fraction=kf,
                                                min_stake=0.0035, max_stake=max_stake, observe_max_stake=observe,
                                                dd_soft=cfg['dd_soft'][0], dd_mid=cfg['dd_mid'][0], dd_hard=cfg['dd_hard'][0],
                                                dd_soft_cap=caps[0], dd_mid_cap=caps[1], dd_hard_cap=caps[2],
                                                weak_window=weak_window, weak_win_min=0.535 if weak_window == 50 else 0.52,
                                                weak_cap=0.0075 if style != 'aggressive' else 0.009,
                                                vol_cap=0.010 if style == 'conservative' else 0.0125,
                                                high_p=0.595 if style != 'aggressive' else 0.585,
                                                risk_on_p=0.575 if style == 'conservative' else 0.565,
                                                update_lag=2,
                                            ))
    return out


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
    if m365.trades < b365.trades * 0.95:
        reasons.append('365交易数未保留95%')
    if off is not None and boff is not None:
        if (off.actualPriceDynamicPnl or off.compoundPnl) < (boff.actualPriceDynamicPnl or boff.compoundPnl):
            reasons.append('官方实际价格动态回放盈亏低于061')
        if off.maxDrawdown > boff.maxDrawdown * 1.10:
            reasons.append('官方实际价格动态回放回撤明显更差')
    return len(reasons) == 0, reasons


def random_label_audit(data: Dict[str, Any], policies: List[V3Policy], b180: WindowMetrics, b365: WindowMetrics) -> Dict[str, Any]:
    total = 0
    sampled = min(20, len(policies))
    by_seed = []
    for seed in [20260512, 20260513, 20260514]:
        rng = np.random.default_rng(seed)
        tr180 = data['train180'].copy(); tr365 = data['train365'].copy(); val180 = data['val180'].copy(); val365 = data['val365'].copy()
        for df in [tr180, tr365, val180, val365]:
            df['won'] = rng.permutation(df['won'].to_numpy())
        strict = 0
        best365 = -1e18
        for pol in policies[:sampled]:
            m180 = simulate_v3(tr180, val180, pol)
            m365 = simulate_v3(tr365, val365, pol)
            if m180.compoundPnl >= b180.compoundPnl * 1.10 and m365.compoundPnl >= b365.compoundPnl * 1.20 and m180.maxDrawdown <= b180.maxDrawdown * 1.05 and m365.maxDrawdown <= b365.maxDrawdown * 1.05:
                strict += 1
            best365 = max(best365, m365.compoundPnl)
        total += strict
        by_seed.append({'seed': seed, 'strictPassCount': strict, 'bestRandom365Pnl': best365})
    return {'passed': total == 0, 'strictPassCount': total, 'sampledPolicies': sampled, 'bySeed': by_seed}


def feature_shift_audit(data: Dict[str, Any], policy: V3Policy) -> Dict[str, Any]:
    val = data['val365'].copy()
    cols = [c for c in policy.dims if c in val.columns]
    for c in cols:
        val[c] = val[c].shift(1).fillna(val[c].iloc[0])
    normal = simulate_v3(data['train365'], data['val365'], policy)
    shifted = simulate_v3(data['train365'], val, policy)
    return {'passed': normal.setHash != shifted.setHash or abs(normal.compoundPnl - shifted.compoundPnl) > 1e-9, 'policy': policy.name, 'shiftedColumns': cols, 'normalHash': normal.setHash, 'shiftedHash': shifted.setHash, 'normal365Pnl': normal.compoundPnl, 'shifted365Pnl': shifted.compoundPnl}


def load_prior_summary(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def run() -> None:
    data = v2.prepare_data()
    b180 = v2.fixed_metrics(data['val180'])
    b365 = v2.fixed_metrics(data['val365'])
    off_base = v2.fixed_metrics(data['official_rows'], actual_mode=True) if not data['official_rows'].empty else None
    baseline_ok = (b180.trades == v2.BASELINE_061['180d']['trades'] and round(b180.compoundPnl, 6) == round(v2.BASELINE_061['180d']['compoundPnl'], 6) and b365.trades == v2.BASELINE_061['365d']['trades'] and round(b365.compoundPnl, 6) == round(v2.BASELINE_061['365d']['compoundPnl'], 6))
    policies = policy_grid()
    audit = {'baselineReproductionPassed': baseline_ok, 'baseline180': metrics_to_dict(b180), 'baseline365': metrics_to_dict(b365), 'officialBaseline': metrics_to_dict(off_base) if off_base else None, 'policyCount': len(policies)}
    if baseline_ok:
        audit['randomLabel'] = random_label_audit(data, policies, b180, b365)
        audit['featureShift'] = feature_shift_audit(data, next(p for p in policies if '1h_vol_bucket' in p.dims and '4h_vol_bucket' in p.dims))
        audit['passed'] = bool(audit['randomLabel']['passed'] and audit['featureShift']['passed'])
    else:
        audit['passed'] = False
    v2.write_json(OUT_AUDIT_JSON, audit)
    OUT_AUDIT_MD.write_text('# 061 风险预算动态仓位 V3 审计\n\n' + f'- 061基线复现: {"通过" if baseline_ok else "失败"}\n- 随机标签: {"通过" if audit.get("randomLabel",{}).get("passed") else "失败"}\n- 特征错位: {"通过" if audit.get("featureShift",{}).get("passed") else "失败"}\n- 候选数: {len(policies)}\n', encoding='utf-8')
    if not audit['passed']:
        raise SystemExit('audit failed')

    rows = []
    for idx, pol in enumerate(policies, 1):
        m180 = simulate_v3(data['train180'], data['val180'], pol)
        m365 = simulate_v3(data['train365'], data['val365'], pol)
        score = ((m180.compoundPnl - b180.compoundPnl) / abs(b180.compoundPnl) * 0.25 + (m365.compoundPnl - b365.compoundPnl) / abs(b365.compoundPnl) * 0.45 + (b365.maxDrawdown - m365.maxDrawdown) / b365.maxDrawdown * 0.30)
        rows.append({'rankScore': score, 'candidateId': digest([pol.name, m180.setHash, m365.setHash]), 'policy': asdict(pol), 'strictPass': False, 'failReasons': [], 'metrics': {'180d': metrics_to_dict(m180), '365d': metrics_to_dict(m365), 'officialActualReplay': None}})
        if idx % 100 == 0:
            v2.write_json(OUT_LEADERBOARD_JSON, sorted(rows, key=lambda r: r['rankScore'], reverse=True)[:200])
    ranked = sorted(rows, key=lambda r: r['rankScore'], reverse=True)
    if off_base is not None and not data['official_rows'].empty:
        for r in ranked[:80]:
            pol = V3Policy(**r['policy'])
            pol_off = copy.copy(pol)
            object.__setattr__(pol_off, 'use_actual_price', True)
            off = simulate_v3(data['official_train'], data['official_rows'], pol_off, actual_mode=True)
            r['metrics']['officialActualReplay'] = metrics_to_dict(off)
            passed, reasons = pass_gate(WindowMetrics(**r['metrics']['180d']), WindowMetrics(**r['metrics']['365d']), off, b180, b365, off_base)
            r['strictPass'] = passed
            r['failReasons'] = reasons
            r['rankScore'] += ((off.actualPriceDynamicPnl - off_base.actualPriceDynamicPnl) / max(abs(off_base.actualPriceDynamicPnl), 1) * 0.10 if off.actualPriceDynamicPnl is not None and off_base.actualPriceDynamicPnl is not None else 0.0)
        ranked = sorted(rows, key=lambda r: r['rankScore'], reverse=True)
    repeat = []
    for r in ranked[:10]:
        pol = V3Policy(**r['policy'])
        a180 = simulate_v3(data['train180'], data['val180'], pol)
        a365 = simulate_v3(data['train365'], data['val365'], pol)
        repeat.append({'candidateId': r['candidateId'], 'passed': a180.setHash == r['metrics']['180d']['setHash'] and a365.setHash == r['metrics']['365d']['setHash'], 'repeat180Hash': a180.setHash, 'repeat365Hash': a365.setHash})
    strict = [r for r in ranked if r['strictPass']]
    verdict = {'baseline': {'180d': metrics_to_dict(b180), '365d': metrics_to_dict(b365), 'officialActualReplay': metrics_to_dict(off_base) if off_base else None}, 'bestOverall': ranked[0] if ranked else None, 'bestStrictPass': strict[0] if strict else None, 'strictPassCount': len(strict), 'repeatChecks': repeat, 'auditPassed': audit['passed'], 'decision': 'FOUND_RISK_BUDGET_CANDIDATE_FOR_SHADOW_ONLY' if strict else 'NO_MID_IMPROVEMENT_CANDIDATE; keep live 061'}
    v2.write_json(OUT_LEADERBOARD_JSON, ranked[:500])
    v2.write_json(OUT_VERDICT_JSON, verdict)

    def md_row(name: str, window: str, m: WindowMetrics) -> str:
        return f'|{name}|{window}|{m.trades}|{m.wins}/{m.losses}|{pct(m.winRate)}|{m.avgStakeMultiplier:.3f}x|{money(m.compoundPnl)}|{money(m.maxDrawdown)}|{m.returnDrawdown:.2f}|'
    lines = ['# 061 风险预算动态仓位 V3 对比', '', '|配置|窗口|交易数|胜/负|胜率|平均仓位倍数|盈亏|最大回撤|收益回撤比|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    lines.append(md_row('当前061固定1%', '180天', b180)); lines.append(md_row('当前061固定1%', '365天', b365))
    if ranked:
        lines.append(md_row('V3综合最强', '180天', WindowMetrics(**ranked[0]['metrics']['180d']))); lines.append(md_row('V3综合最强', '365天', WindowMetrics(**ranked[0]['metrics']['365d'])))
    if strict:
        lines.append(md_row('V3严格通过最强', '180天', WindowMetrics(**strict[0]['metrics']['180d']))); lines.append(md_row('V3严格通过最强', '365天', WindowMetrics(**strict[0]['metrics']['365d'])))
    lines += ['', '## 官方实际订单回放', '', '|配置|订单数|胜/负|胜率|动态重算盈亏|同批订单原始金额盈亏(非V3结果)|最大回撤|平均买价|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    if off_base:
        lines.append(f'|当前061固定1%|{off_base.trades}|{off_base.wins}/{off_base.losses}|{pct(off_base.winRate)}|{money(off_base.actualPriceDynamicPnl or off_base.compoundPnl)}|{money(off_base.originalLiveAmountPnl or 0.0)}|{money(off_base.maxDrawdown)}|{(off_base.weightedAvgBuyPrice or 0):.4f}|')
    if ranked and ranked[0]['metrics'].get('officialActualReplay'):
        mo = WindowMetrics(**ranked[0]['metrics']['officialActualReplay'])
        lines.append(f'|V3综合最强|{mo.trades}|{mo.wins}/{mo.losses}|{pct(mo.winRate)}|{money(mo.actualPriceDynamicPnl or mo.compoundPnl)}|{money(mo.originalLiveAmountPnl or 0.0)}|{money(mo.maxDrawdown)}|{(mo.weightedAvgBuyPrice or 0):.4f}|')
    if strict and strict[0]['metrics'].get('officialActualReplay'):
        mo = WindowMetrics(**strict[0]['metrics']['officialActualReplay'])
        lines.append(f'|V3严格通过最强|{mo.trades}|{mo.wins}/{mo.losses}|{pct(mo.winRate)}|{money(mo.actualPriceDynamicPnl or mo.compoundPnl)}|{money(mo.originalLiveAmountPnl or 0.0)}|{money(mo.maxDrawdown)}|{(mo.weightedAvgBuyPrice or 0):.4f}|')
    lines += ['', f'严格通过候选数: {len(strict)}', f'结论: {verdict["decision"]}']
    OUT_COMPARE_MD.write_text('\n'.join(lines), encoding='utf-8')
    v2.write_json(OUT_COMPARE_JSON, verdict)
    OUT_LEADERBOARD_MD.write_text('# 061 风险预算动态仓位 V3 排行榜\n\n' + '\n'.join([f"{i+1}. `{r['candidateId']}` score={r['rankScore']:.4f} pass={r['strictPass']} 365pnl={r['metrics']['365d']['compoundPnl']:.2f} dd={r['metrics']['365d']['maxDrawdown']:.2f}" for i, r in enumerate(ranked[:80])]), encoding='utf-8')
    OUT_VERDICT_MD.write_text('# 061 风险预算动态仓位 V3 唯一结论\n\n' + f'- 严格通过候选数: `{len(strict)}`\n- 结论: `{verdict["decision"]}`\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    run()
