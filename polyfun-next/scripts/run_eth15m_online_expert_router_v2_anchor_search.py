#!/usr/bin/env python3
"""Research-only ETH15m online expert router V2: 061 anchored stake router.

Does not touch live trading config.  The strategy keeps 061 as the direction
and default trade set, while online experts only scale stake or rarely skip.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path('/Users/mac/polyfun')
SCRIPTS = ROOT / 'polyfun-next' / 'scripts'
REPORTS = ROOT / 'reports'
REPORTS.mkdir(parents=True, exist_ok=True)
V1_PATH = SCRIPTS / 'run_eth15m_online_expert_router_search.py'
V2_SOURCE_PATH = SCRIPTS / 'run_top159_061_bayesian_sizer_v2_search.py'

BUY_PRICE = 0.55
INITIAL_FUNDS = 850.0
BASE_STAKE = 0.01
UPDATE_LAG = 2

OUT_AUDIT_JSON = REPORTS / 'eth15m_online_expert_router_v2_anchor_bug_audit_latest.json'
OUT_AUDIT_MD = REPORTS / 'eth15m_online_expert_router_v2_anchor_bug_audit_latest.md'
OUT_LEADERBOARD_JSON = REPORTS / 'eth15m_online_expert_router_v2_anchor_leaderboard_latest.json'
OUT_LEADERBOARD_MD = REPORTS / 'eth15m_online_expert_router_v2_anchor_leaderboard_latest.md'
OUT_COMPARE_JSON = REPORTS / 'eth15m_online_expert_router_v2_anchor_061_compare_latest.json'
OUT_COMPARE_MD = REPORTS / 'eth15m_online_expert_router_v2_anchor_061_compare_latest.md'
OUT_VERDICT_JSON = REPORTS / 'eth15m_online_expert_router_v2_anchor_unique_verdict_latest.json'
OUT_VERDICT_MD = REPORTS / 'eth15m_online_expert_router_v2_anchor_unique_verdict_latest.md'


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

v1 = _load(V1_PATH, 'online_router_v1_for_anchor')
v2src = _load(V2_SOURCE_PATH, 'router_v2_source_for_anchor')


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def digest_rows(parts: Iterable[Any]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode('utf-8')); h.update(b'|')
    return h.hexdigest()[:16]


def pct(x: float) -> str:
    return f'{x * 100:.2f}%'


def money(x: float) -> str:
    return f'{x:,.2f}'


@dataclass(frozen=True)
class AnchorPolicy:
    name: str
    method: str
    learning_rate: float
    decay: float
    quality_center: float
    up_threshold: float
    down_threshold: float
    skip_threshold: float
    stake_low: float
    stake_normal: float
    stake_high: float
    stake_very_high: float
    dd_lock: float
    dd_lock_cap: float
    update_lag: int = UPDATE_LAG
    rolling_window: int = 50
    bayes_prior: float = 40.0
    min_skip_bad_share: float = 0.62


@dataclass
class AnchorMetrics:
    rows: int
    trades: int
    skipped: int
    wins: int
    losses: int
    winRate: float
    avgStakePct: float
    avgStakeMultiplier: float
    upsizeCount: int
    downsizeCount: int
    skipCount: int
    compoundPnl: float
    endingFunds: float
    maxDrawdown: float
    returnDrawdown: float
    retentionRate: float
    setHash: str
    officialActualPnl: Optional[float] = None
    officialBasePnl: Optional[float] = None
    officialWeightedAvgBuyPrice: Optional[float] = None
    officialBlockedWinners: Optional[int] = None
    officialBlockedLosers: Optional[int] = None


def metrics_to_dict(m: AnchorMetrics) -> Dict[str, Any]:
    return asdict(m)


class QualityWeights:
    def __init__(self, n: int, pol: AnchorPolicy):
        self.n = n; self.pol = pol
        self.logw = np.zeros(n, dtype=float)
        self.wins = np.full(n, pol.bayes_prior * 0.575, dtype=float)
        self.counts = np.full(n, pol.bayes_prior, dtype=float)
        self.roll = [deque(maxlen=pol.rolling_window) for _ in range(n)]

    def weights(self) -> np.ndarray:
        if self.pol.method == 'hedge':
            x = self.logw - np.max(self.logw)
            return np.exp(x)
        if self.pol.method == 'bayes':
            return np.maximum(self.wins / np.maximum(self.counts, 1e-9), 1e-6)
        vals = []
        for q in self.roll:
            vals.append(0.575 if not q else max(0.01, 0.575 + float(np.mean(q)) * 0.18))
        return np.asarray(vals, dtype=float)

    def update(self, actions: np.ndarray, won: bool) -> None:
        # action codes from V1: 0=skip/no opinion, 1=follow expert, 2=alt expert.
        rewards = np.zeros(self.n, dtype=float)
        rewards[(actions == 1) & won] = 1.0 / BUY_PRICE - 1.0
        rewards[(actions == 1) & (not won)] = -1.0
        rewards[(actions == 2) & (not won)] = 1.0 / BUY_PRICE - 1.0
        rewards[(actions == 2) & won] = -1.0
        if self.pol.method == 'hedge':
            self.logw = self.logw * self.pol.decay + self.pol.learning_rate * rewards
        elif self.pol.method == 'bayes':
            self.wins *= self.pol.decay; self.counts *= self.pol.decay
            non_skip = actions != 0
            correct = ((actions == 1) & won) | ((actions == 2) & (not won))
            self.wins += correct.astype(float); self.counts += non_skip.astype(float)
        else:
            for i, r in enumerate(rewards):
                self.roll[i].append(float(r))


def quality_from_actions(actions: np.ndarray, weights: np.ndarray) -> Tuple[float, float, float, int]:
    follow = float(weights[actions == 1].sum())
    alt = float(weights[actions == 2].sum())
    active = int(np.count_nonzero(actions != 0))
    total = max(follow + alt, 1e-12)
    quality = follow / total if active else 0.575
    bad_share = alt / total if active else 0.0
    return quality, follow, bad_share, active


def policy_grid() -> List[AnchorPolicy]:
    out: List[AnchorPolicy] = []
    stake_sets = {
        'conservative': (0.0065, 0.0100, 0.0110, 0.0120),
        'balanced': (0.0050, 0.0100, 0.0120, 0.0135),
        'aggressive': (0.0050, 0.0100, 0.0135, 0.0150),
    }
    for method in ['hedge', 'bayes', 'rolling']:
        for lr in ([0.06, 0.12, 0.20] if method == 'hedge' else [0.06]):
            for decay in ([0.995, 1.0] if method in {'hedge', 'bayes'} else [1.0]):
                for up_t in [0.58, 0.60, 0.62]:
                    for down_t in [0.48, 0.50, 0.52]:
                        for skip_t in [0.38, 0.42, 0.45]:
                            if skip_t >= down_t:
                                continue
                            for dd_lock in [0.03, 0.05, 0.07]:
                                for label, stakes in stake_sets.items():
                                    for rw in ([20, 50, 100] if method == 'rolling' else [50]):
                                        nm = digest_rows([method, lr, decay, up_t, down_t, skip_t, dd_lock, label, rw])
                                        out.append(AnchorPolicy(
                                            name=f'anchor_v2_{nm}', method=method, learning_rate=lr, decay=decay,
                                            quality_center=0.575, up_threshold=up_t, down_threshold=down_t, skip_threshold=skip_t,
                                            stake_low=stakes[0], stake_normal=stakes[1], stake_high=stakes[2], stake_very_high=stakes[3],
                                            dd_lock=dd_lock, dd_lock_cap=0.0100, rolling_window=rw,
                                        ))
    # Representative but broad enough: 90 per method if available.
    selected: List[AnchorPolicy] = []
    for method in ['hedge', 'bayes', 'rolling']:
        chunk = [p for p in out if p.method == method]
        if len(chunk) <= 90:
            selected.extend(chunk)
        else:
            idxs = np.linspace(0, len(chunk) - 1, 90, dtype=int)
            selected.extend([chunk[int(i)] for i in idxs])
    return selected


def stake_from_quality(pol: AnchorPolicy, quality: float, bad_share: float, active: int, funds: float, peak: float) -> Tuple[float, str]:
    dd = (peak - funds) / peak if peak > 0 else 0.0
    if active > 0 and quality <= pol.skip_threshold and bad_share >= pol.min_skip_bad_share:
        return 0.0, 'skip'
    if quality >= pol.up_threshold + 0.035:
        stake = pol.stake_very_high; reason = 'very_high'
    elif quality >= pol.up_threshold:
        stake = pol.stake_high; reason = 'high'
    elif quality <= pol.down_threshold:
        stake = pol.stake_low; reason = 'low'
    else:
        stake = pol.stake_normal; reason = 'normal'
    if dd >= pol.dd_lock:
        stake = min(stake, pol.dd_lock_cap)
        reason = 'dd_lock_' + reason
    return stake, reason


def simulate_anchor(train: pd.DataFrame, val: pd.DataFrame, train_actions: np.ndarray, val_actions: np.ndarray, pol: AnchorPolicy, *, official_actual: bool = False) -> AnchorMetrics:
    train = train.sort_values('dt').reset_index(drop=True)
    val = val.sort_values('dt').reset_index(drop=True)
    state = QualityWeights(train_actions.shape[1], pol)
    for i, row in train.iterrows():
        state.update(train_actions[i], bool(row['won']))

    funds = INITIAL_FUNDS; peak = funds; max_dd = 0.0
    trades = wins = losses = skipped = up = down = 0
    stake_pcts: List[float] = []
    original_pnl = 0.0 if official_actual else None
    weighted_cost = weighted_amt = 0.0
    blocked_w = blocked_l = 0
    parts: List[Any] = []
    pending: deque = deque()

    def flush_until(idx: int):
        while pending and pending[0][0] <= idx:
            _, acts, won = pending.popleft()
            state.update(acts, bool(won))

    for i, row in val.iterrows():
        flush_until(i)
        weights = state.weights()
        quality, _, bad_share, active = quality_from_actions(val_actions[i], weights)
        stake_pct, reason = stake_from_quality(pol, quality, bad_share, active, funds, peak)
        pending.append((i + pol.update_lag, val_actions[i], bool(row['won'])))
        won = bool(row['won'])

        if stake_pct <= 0:
            skipped += 1
            if won: blocked_w += 1
            else: blocked_l += 1
            parts.append((row.get('market_slug', i), reason, round(quality, 6), round(bad_share, 6), won))
            continue

        if official_actual:
            base_cost = float(row.get('actual_cost', 0.0) or 0.0)
            base_pnl = float(row.get('actual_pnl', 0.0) or 0.0)
            scale = stake_pct / BASE_STAKE
            pnl = base_pnl * scale
            if original_pnl is not None:
                original_pnl += base_pnl
            if base_cost > 0 and 'actual_avg_price' in row and pd.notna(row['actual_avg_price']):
                weighted_cost += base_cost * scale * float(row['actual_avg_price'])
                weighted_amt += base_cost * scale
        else:
            stake = funds * stake_pct
            pnl = stake * (1.0 / BUY_PRICE - 1.0) if won else -stake
        funds += pnl
        peak = max(peak, funds)
        max_dd = max(max_dd, peak - funds)
        trades += 1
        if won: wins += 1
        else: losses += 1
        stake_pcts.append(stake_pct)
        if stake_pct > BASE_STAKE * 1.05: up += 1
        if stake_pct < BASE_STAKE * 0.95: down += 1
        parts.append((row.get('market_slug', i), reason, round(quality, 6), round(bad_share, 6), round(stake_pct, 5), won, round(pnl, 6)))

    pnl = funds - INITIAL_FUNDS
    avg = float(np.mean(stake_pcts)) if stake_pcts else 0.0
    return AnchorMetrics(
        rows=int(len(val)), trades=int(trades), skipped=int(skipped), wins=int(wins), losses=int(losses),
        winRate=float(wins / trades if trades else 0.0), avgStakePct=avg, avgStakeMultiplier=float(avg / BASE_STAKE if BASE_STAKE else 0.0),
        upsizeCount=int(up), downsizeCount=int(down), skipCount=int(skipped), compoundPnl=float(pnl), endingFunds=float(funds),
        maxDrawdown=float(max_dd), returnDrawdown=float(pnl / max_dd) if max_dd else 0.0, retentionRate=float(trades / len(val) if len(val) else 0.0),
        setHash=digest_rows(parts), officialActualPnl=float(pnl) if official_actual else None,
        officialBasePnl=float(original_pnl) if original_pnl is not None else None,
        officialWeightedAvgBuyPrice=float(weighted_cost / weighted_amt) if weighted_amt > 0 else None,
        officialBlockedWinners=int(blocked_w) if official_actual else None,
        officialBlockedLosers=int(blocked_l) if official_actual else None,
    )


def baseline_metrics(df: pd.DataFrame) -> AnchorMetrics:
    m = v2src.fixed_metrics(df)
    return AnchorMetrics(m.rows, m.trades, 0, m.wins, m.losses, m.winRate, BASE_STAKE, 1.0, 0, 0, 0, m.compoundPnl, m.endingFunds, m.maxDrawdown, m.returnDrawdown, 1.0, m.setHash)


def official_baseline(df: pd.DataFrame) -> AnchorMetrics:
    funds = INITIAL_FUNDS; peak = funds; max_dd = 0.0; wins = losses = 0; pnl_sum = 0.0; cost = amt = 0.0; parts = []
    for i, row in df.sort_values('dt').reset_index(drop=True).iterrows():
        won = bool(row['won']); pnl = float(row.get('actual_pnl', 0.0) or 0.0); pnl_sum += pnl; funds += pnl
        peak = max(peak, funds); max_dd = max(max_dd, peak - funds); wins += int(won); losses += int(not won)
        c = float(row.get('actual_cost', 0.0) or 0.0)
        if c > 0 and 'actual_avg_price' in row and pd.notna(row['actual_avg_price']):
            cost += c * float(row['actual_avg_price']); amt += c
        parts.append((row.get('market_slug', i), won, round(pnl, 6)))
    trades = wins + losses
    return AnchorMetrics(len(df), trades, 0, wins, losses, wins / trades if trades else 0.0, BASE_STAKE, 1.0, 0, 0, 0, pnl_sum, funds, max_dd, pnl_sum / max_dd if max_dd else 0.0, 1.0, digest_rows(parts), pnl_sum, pnl_sum, cost / amt if amt else None, 0, 0)


def pass_gate(m180: AnchorMetrics, m365: AnchorMetrics, off: Optional[AnchorMetrics], b180: AnchorMetrics, b365: AnchorMetrics, boff: Optional[AnchorMetrics]) -> Tuple[bool, List[str]]:
    reasons = []
    if m365.compoundPnl < b365.compoundPnl * 1.20: reasons.append('365盈亏未高于061 20%')
    if m180.compoundPnl < b180.compoundPnl * 1.10: reasons.append('180盈亏未高于061 10%')
    if m365.maxDrawdown > b365.maxDrawdown * 1.05: reasons.append('365回撤高于061超过5%')
    if m180.maxDrawdown > b180.maxDrawdown * 1.05: reasons.append('180回撤高于061超过5%')
    if m365.retentionRate < 0.90: reasons.append('365保留率低于90%')
    if off and boff:
        if off.compoundPnl < boff.compoundPnl: reasons.append('官方实际订单动态重算盈亏低于061')
        if off.maxDrawdown > boff.maxDrawdown * 1.10: reasons.append('官方实际订单动态重算回撤明显更差')
    return len(reasons) == 0, reasons


def audit_random(data: Dict[str, Any], actions: Dict[str, np.ndarray], policies: Sequence[AnchorPolicy], b180: AnchorMetrics, b365: AnchorMetrics) -> Dict[str, Any]:
    total = 0; by_seed = []
    sample = policies[:min(12, len(policies))]
    for seed in [20260513, 20260514, 20260515]:
        rng = np.random.default_rng(seed)
        tr180 = data['train180'].copy(); tr365 = data['train365'].copy(); v180 = data['val180'].copy(); v365 = data['val365'].copy()
        for df in [tr180, tr365, v180, v365]: df['won'] = rng.permutation(df['won'].to_numpy())
        strict = 0; best = -1e18
        for p in sample:
            m180 = simulate_anchor(tr180, v180, actions['train180'], actions['val180'], p)
            m365 = simulate_anchor(tr365, v365, actions['train365'], actions['val365'], p)
            ok, _ = pass_gate(m180, m365, None, b180, b365, None); strict += int(ok); best = max(best, m365.compoundPnl)
        total += strict; by_seed.append({'seed': seed, 'strictPassCount': strict, 'bestRandom365Pnl': best})
    return {'passed': total == 0, 'strictPassCount': total, 'sampledPolicies': len(sample), 'bySeed': by_seed}


def audit_shift(data: Dict[str, Any], experts: Sequence[Any], actions: Dict[str, np.ndarray], policy: AnchorPolicy) -> Dict[str, Any]:
    val = data['val365'].copy()
    for c in [c for c in ['score15','1h_trend_bucket','4h_pos_bucket','hour_bucket','cluster_hits'] if c in val.columns]:
        val[c] = val[c].shift(1).fillna(val[c].iloc[0])
    shifted_actions = v1.precompute_actions(val, experts)
    normal = simulate_anchor(data['train365'], data['val365'], actions['train365'], actions['val365'], policy)
    shifted = simulate_anchor(data['train365'], val, actions['train365'], shifted_actions, policy)
    return {'passed': normal.setHash != shifted.setHash or abs(normal.compoundPnl - shifted.compoundPnl) > 1e-9, 'normalHash': normal.setHash, 'shiftedHash': shifted.setHash, 'normal365Pnl': normal.compoundPnl, 'shifted365Pnl': shifted.compoundPnl}


def md_table(verdict: Dict[str, Any]) -> str:
    lines = ['# ETH15m 在线专家路由器 V2：061锚定加减仓对比','']
    lines += ['|配置|窗口|交易数|胜/负|胜率|平均仓位|加仓|降仓|跳过|盈亏|最大回撤|收益回撤比|保留率|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, obj in [('当前061', verdict['baseline']), ('V1最佳路由器', verdict.get('v1Best')), ('V2最佳锚定路由', verdict.get('bestOverall')), ('V2严格候选', verdict.get('bestStrictPass'))]:
        if not obj: continue
        metrics = obj.get('metrics', obj)
        for w in ['180d','365d']:
            m = metrics[w]
            lines.append(f"|{name}|{w}|{m['trades']}|{m['wins']}/{m['losses']}|{pct(m['winRate'])}|{m.get('avgStakeMultiplier',1):.3f}x|{m.get('upsizeCount',0)}|{m.get('downsizeCount',0)}|{m.get('skipCount',m.get('skipped',0))}|{money(m['compoundPnl'])}|{money(m['maxDrawdown'])}|{m['returnDrawdown']:.2f}|{pct(m['retentionRate'])}|")
    lines += ['', '## 官方实际订单回放', '', '|配置|订单数|保留订单|胜/负|胜率|原始061官方盈亏|动态重算盈亏|最大回撤|平均买价|拦截赢家|拦截输单|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, obj in [('当前061', verdict['baseline'].get('officialActualReplay')), ('V2最佳锚定路由', verdict.get('bestOverall',{}).get('metrics',{}).get('officialActualReplay') if verdict.get('bestOverall') else None), ('V2严格候选', verdict.get('bestStrictPass',{}).get('metrics',{}).get('officialActualReplay') if verdict.get('bestStrictPass') else None)]:
        if not obj: continue
        lines.append(f"|{name}|{obj['rows']}|{obj['trades']}|{obj['wins']}/{obj['losses']}|{pct(obj['winRate'])}|{money(obj.get('officialBasePnl') or 0)}|{money(obj['compoundPnl'])}|{money(obj['maxDrawdown'])}|{obj.get('officialWeightedAvgBuyPrice') or '-'}|{obj.get('officialBlockedWinners')}|{obj.get('officialBlockedLosers')}|")
    lines += ['', f"严格通过候选数: {verdict['strictPassCount']}", f"结论: {verdict['decision']}"]
    return '\n'.join(lines)


def run() -> None:
    data = v2src.prepare_data()
    all_df = pd.concat([data['train365'], data['val365']], ignore_index=True)
    experts = v1.build_experts(all_df)
    actions = {k: v1.precompute_actions(data[k], experts) if not data[k].empty else np.zeros((0, len(experts)), dtype=np.int8) for k in ['train180','val180','train365','val365','official_train','official_rows']}
    policies = policy_grid()
    b180 = baseline_metrics(data['val180']); b365 = baseline_metrics(data['val365']); boff = official_baseline(data['official_rows']) if not data['official_rows'].empty else None
    baseline_ok = b180.trades == v2src.BASELINE_061['180d']['trades'] and round(b180.compoundPnl,6) == round(v2src.BASELINE_061['180d']['compoundPnl'],6) and b365.trades == v2src.BASELINE_061['365d']['trades'] and round(b365.compoundPnl,6) == round(v2src.BASELINE_061['365d']['compoundPnl'],6)
    audit = {'baselineReproductionPassed': baseline_ok, 'expertCount': len(experts), 'policyCount': len(policies), 'baseline180': asdict(b180), 'baseline365': asdict(b365), 'officialBaseline': asdict(boff) if boff else None}
    if baseline_ok:
        audit['randomLabel'] = audit_random(data, actions, policies, b180, b365)
        audit['featureShift'] = audit_shift(data, experts, actions, policies[0])
        audit['passed'] = bool(audit['randomLabel']['passed'] and audit['featureShift']['passed'])
    else:
        audit['passed'] = False
    write_json(REPORTS / 'eth15m_online_expert_router_v2_anchor_bug_audit_latest.json', audit)
    (REPORTS / 'eth15m_online_expert_router_v2_anchor_bug_audit_latest.md').write_text('# ETH15m 在线专家路由器 V2 审计\n\n' + f"- 061复现: {'通过' if baseline_ok else '失败'}\n- 专家数: `{len(experts)}`\n- 策略数: `{len(policies)}`\n- 随机标签: {'通过' if audit.get('randomLabel',{}).get('passed') else '失败'}\n- 特征错位: {'通过' if audit.get('featureShift',{}).get('passed') else '失败'}\n", encoding='utf-8')
    if not audit['passed']:
        raise SystemExit('audit failed')

    rows = []
    for idx, p in enumerate(policies, 1):
        m180 = simulate_anchor(data['train180'], data['val180'], actions['train180'], actions['val180'], p)
        m365 = simulate_anchor(data['train365'], data['val365'], actions['train365'], actions['val365'], p)
        score = (m365.compoundPnl - b365.compoundPnl) / max(abs(b365.compoundPnl),1) * 0.45 + (m180.compoundPnl - b180.compoundPnl) / max(abs(b180.compoundPnl),1) * 0.25 + (b365.maxDrawdown - m365.maxDrawdown) / max(b365.maxDrawdown,1) * 0.20 + (m365.retentionRate - 0.95) * 0.10
        rows.append({'rankScore': score, 'candidateId': digest_rows([p.name, m180.setHash, m365.setHash]), 'policy': asdict(p), 'strictPass': False, 'failReasons': [], 'metrics': {'180d': asdict(m180), '365d': asdict(m365), 'officialActualReplay': None}})
        if idx % 50 == 0:
            write_json(REPORTS / 'eth15m_online_expert_router_v2_anchor_leaderboard_latest.json', sorted(rows, key=lambda r:r['rankScore'], reverse=True)[:300])
    ranked = sorted(rows, key=lambda r:r['rankScore'], reverse=True)
    if boff is not None:
        for r in ranked[:50]:
            p = AnchorPolicy(**r['policy'])
            off = simulate_anchor(data['official_train'], data['official_rows'], actions['official_train'], actions['official_rows'], p, official_actual=True)
            r['metrics']['officialActualReplay'] = asdict(off)
            ok, reasons = pass_gate(AnchorMetrics(**r['metrics']['180d']), AnchorMetrics(**r['metrics']['365d']), off, b180, b365, boff)
            r['strictPass'] = ok; r['failReasons'] = reasons
            r['rankScore'] += (off.compoundPnl - boff.compoundPnl) / max(abs(boff.compoundPnl), 1) * 0.10
        ranked = sorted(rows, key=lambda r:r['rankScore'], reverse=True)
    repeat = []
    for r in ranked[:10]:
        p = AnchorPolicy(**r['policy'])
        a180 = simulate_anchor(data['train180'], data['val180'], actions['train180'], actions['val180'], p)
        a365 = simulate_anchor(data['train365'], data['val365'], actions['train365'], actions['val365'], p)
        repeat.append({'candidateId': r['candidateId'], 'passed': a180.setHash == r['metrics']['180d']['setHash'] and a365.setHash == r['metrics']['365d']['setHash'], 'repeat180Hash': a180.setHash, 'repeat365Hash': a365.setHash})
    strict = [r for r in ranked if r['strictPass']]
    v1_verdict = {}
    try:
        v1_verdict = json.loads((REPORTS / 'eth15m_online_expert_router_unique_verdict_latest.json').read_text())
    except Exception:
        v1_verdict = {}
    verdict = {'auditPassed': audit['passed'], 'baseline': {'180d': asdict(b180), '365d': asdict(b365), 'officialActualReplay': asdict(boff) if boff else None}, 'v1Best': v1_verdict.get('bestOverall'), 'bestOverall': ranked[0] if ranked else None, 'bestStrictPass': strict[0] if strict else None, 'strictPassCount': len(strict), 'repeatChecks': repeat, 'decision': 'FOUND_ANCHOR_EXPERT_SIZER_FOR_SHADOW_ONLY' if strict else 'NO_ANCHOR_EXPERT_SIZER_CANDIDATE; keep live 061'}
    write_json(REPORTS / 'eth15m_online_expert_router_v2_anchor_leaderboard_latest.json', ranked[:500])
    write_json(REPORTS / 'eth15m_online_expert_router_v2_anchor_unique_verdict_latest.json', verdict)
    write_json(REPORTS / 'eth15m_online_expert_router_v2_anchor_061_compare_latest.json', verdict)
    (REPORTS / 'eth15m_online_expert_router_v2_anchor_061_compare_latest.md').write_text(md_table(verdict), encoding='utf-8')
    (REPORTS / 'eth15m_online_expert_router_v2_anchor_leaderboard_latest.md').write_text('# ETH15m 在线专家路由器 V2 排行榜\n\n' + '\n'.join([f"{i+1}. `{r['candidateId']}` score={r['rankScore']:.4f} pass={r['strictPass']} 365pnl={r['metrics']['365d']['compoundPnl']:.2f} dd={r['metrics']['365d']['maxDrawdown']:.2f} avgStake={r['metrics']['365d']['avgStakeMultiplier']:.3f}x" for i,r in enumerate(ranked[:100])]), encoding='utf-8')
    (REPORTS / 'eth15m_online_expert_router_v2_anchor_unique_verdict_latest.md').write_text('# ETH15m 在线专家路由器 V2 唯一结论\n\n' + f"- 严格通过候选数: `{len(strict)}`\n- 结论: `{verdict['decision']}`\n", encoding='utf-8')
    print(md_table(verdict))


if __name__ == '__main__':
    run()
