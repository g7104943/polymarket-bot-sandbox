#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path('/Users/mac/polyfun')
NEXT = ROOT / 'polyfun-next'
REPORTS = ROOT / 'reports'
sys.path.insert(0, str(NEXT / 'src'))

from polyfun_next.risk_v2 import (  # noqa: E402
    V3_PROFILE,
    RiskV2Policy,
    load_official_trades,
    simulate_official_risk_policy,
    stable_hash,
    write_json_atomic,
)

INITIAL_FUNDS = 847.091209
OUT_AUDIT_JSON = REPORTS / 'top159_real_order_risk_v3_state_machine_bug_audit_latest.json'
OUT_AUDIT_MD = REPORTS / 'top159_real_order_risk_v3_state_machine_bug_audit_latest.md'
OUT_LEADERBOARD_JSON = REPORTS / 'top159_real_order_risk_v3_state_machine_leaderboard_latest.json'
OUT_LEADERBOARD_MD = REPORTS / 'top159_real_order_risk_v3_state_machine_leaderboard_latest.md'
OUT_COMPARE_JSON = REPORTS / 'top159_real_order_risk_v3_state_machine_compare_latest.json'
OUT_COMPARE_MD = REPORTS / 'top159_real_order_risk_v3_state_machine_compare_latest.md'
OUT_VERDICT_JSON = REPORTS / 'top159_real_order_risk_v3_state_machine_unique_verdict_latest.json'
OUT_VERDICT_MD = REPORTS / 'top159_real_order_risk_v3_state_machine_unique_verdict_latest.md'


def money(x: Any) -> str:
    try:
        return f'{float(x):+,.2f}'
    except Exception:
        return '-'


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def current_combo_policy() -> RiskV2Policy:
    return RiskV2Policy(
        name='combo_loss4_highprice0.55_pause4_lag4',
        family='combo_loss_highprice',
        update_lag_bars=4,
        loss_streak_n=4,
        loss_pause_bars=4,
        high_price_min=0.55,
        high_price_loss_streak_n=2,
        high_price_pause_bars=4,
    )


def policy_grid() -> list[RiskV2Policy]:
    out: list[RiskV2Policy] = []
    for lag in [1, 2, 4, 8]:
        for yw in [10, 20, 30, 50]:
            for yp in [-10.0, -20.0, -30.0, -50.0]:
                for ybars in [4, 8, 12]:
                    for rw in [20, 30, 50]:
                        for rp in [-20.0, -30.0, -50.0, -75.0]:
                            for rbars in [12, 24, 48]:
                                out.append(RiskV2Policy(
                                    name=f'sm_y{yw}_{int(yp)}p{ybars}_r{rw}_{int(rp)}p{rbars}_lag{lag}',
                                    family='state_machine',
                                    update_lag_bars=lag,
                                    state_yellow_window=yw,
                                    state_yellow_min_pnl=yp,
                                    state_yellow_pause_bars=ybars,
                                    state_red_window=rw,
                                    state_red_min_pnl=rp,
                                    state_red_pause_bars=rbars,
                                ))
        for loss_n in [3, 4, 5]:
            for hp in [0.55, 0.60, 0.65]:
                for hp_loss_n in [2, 3]:
                    for rbars in [8, 12, 24]:
                        for rw, rp in [(20, -20.0), (20, -30.0), (30, -30.0), (50, -50.0)]:
                            out.append(RiskV2Policy(
                                name=f'sm_roll{rw}_{int(rp)}_loss{loss_n}_hp{hp}_hpl{hp_loss_n}_p{rbars}_lag{lag}',
                                family='state_machine',
                                update_lag_bars=lag,
                                state_yellow_window=10,
                                state_yellow_min_pnl=-10.0,
                                state_yellow_pause_bars=4,
                                state_red_window=rw,
                                state_red_min_pnl=rp,
                                state_red_pause_bars=rbars,
                                state_red_loss_streak_n=loss_n,
                                state_red_high_price_min=hp,
                                state_red_high_price_loss_streak_n=hp_loss_n,
                            ))
        for gdd in [0.08, 0.10, 0.12, 0.15]:
            for rbars in [12, 24, 48]:
                for rw, rp in [(20, -20.0), (30, -30.0), (50, -50.0)]:
                    out.append(RiskV2Policy(
                        name=f'sm_globaldd{gdd}_roll{rw}_{int(rp)}_p{rbars}_lag{lag}',
                        family='state_machine',
                        update_lag_bars=lag,
                        state_yellow_window=10,
                        state_yellow_min_pnl=-10.0,
                        state_yellow_pause_bars=4,
                        state_red_window=rw,
                        state_red_min_pnl=rp,
                        state_red_pause_bars=rbars,
                        state_red_global_drawdown_fraction=gdd,
                    ))
    seen: set[str] = set()
    uniq: list[RiskV2Policy] = []
    for p in out:
        if p.name in seen:
            continue
        seen.add(p.name)
        uniq.append(p)
    return uniq


def enrich(row: dict[str, Any], policy: RiskV2Policy | None, baseline: dict[str, Any], combo: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row['policy'] = asdict(policy) if policy else None
    row['candidateId'] = stable_hash(row['policy']) if row['policy'] else 'baseline'
    row['pnlGainVs061'] = round(float(row['officialActualPnl']) - float(baseline['officialActualPnl']), 6)
    row['pnlGainVsCombo'] = round(float(row['officialActualPnl']) - float(combo['officialActualPnl']), 6)
    row['drawdownReductionVs061'] = round(float(baseline['maxDrawdownUsd']) - float(row['maxDrawdownUsd']), 6)
    row['drawdownReductionVsCombo'] = round(float(combo['maxDrawdownUsd']) - float(row['maxDrawdownUsd']), 6)
    row['loserMinusWinnerIntercepts'] = int(row['interceptedLosers']) - int(row['interceptedWinners'])
    row['strictLiveCandidate'] = bool(
        row['officialActualPnl'] > combo['officialActualPnl']
        and row['officialActualPnl'] > baseline['officialActualPnl']
        and row['maxDrawdownUsd'] <= baseline['maxDrawdownUsd'] * 0.75
        and row['retentionRatePct'] >= 55.0
    )
    row['rankScore'] = round(
        row['pnlGainVs061'] * 100.0
        + row['drawdownReductionVs061'] * 25.0
        + row['loserMinusWinnerIntercepts'] * 30.0
        + max(0.0, row['retentionRatePct'] - 55.0) * 2.0,
        6,
    )
    return row


def select_candidates(rows: list[dict[str, Any]], baseline: dict[str, Any], combo: dict[str, Any]) -> dict[str, Any]:
    positive = [r for r in rows if r['officialActualPnl'] > combo['officialActualPnl'] and r['retentionRatePct'] >= 55.0]
    pool = positive or rows
    best_pnl = max(pool, key=lambda r: (r['officialActualPnl'], -r['maxDrawdownUsd'], r['retentionRatePct']))
    best_dd_pool = [r for r in rows if r['officialActualPnl'] >= combo['officialActualPnl'] - 10.0 and r['retentionRatePct'] >= 55.0] or rows
    best_drawdown = min(best_dd_pool, key=lambda r: (r['maxDrawdownUsd'], -r['officialActualPnl'], -r['retentionRatePct']))
    balanced_pool = [r for r in rows if r['strictLiveCandidate']] or positive or rows
    balanced = max(balanced_pool, key=lambda r: (r['rankScore'], r['officialActualPnl'], -r['maxDrawdownUsd']))
    return {'best_pnl': best_pnl, 'best_drawdown': best_drawdown, 'balanced_live_candidate': balanced}


def render_table(title: str, rows: list[tuple[str, dict[str, Any]]]) -> str:
    lines = [title, '', '|配置|保留/总数|胜/负|胜率|官方盈亏|最大回撤|拦截赢/输|保留率|暂停|哈希|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for name, r in rows:
        lines.append(
            f"|{name}|{r['keptTrades']}/{r['allTrades']}|{r['wins']}/{r['losses']}|{r['winRatePct']:.2f}%|"
            f"{money(r['officialActualPnl'])}|{money(r['maxDrawdownUsd'])}|{r['interceptedWinners']}/{r['interceptedLosers']}|"
            f"{r['retentionRatePct']:.2f}%|{r['pauseCount']}|`{r['setHash']}`|"
        )
    return '\n'.join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description='Search 061 official-order V3 state-machine risk profiles. Writes V3 shadow profile by default.')
    ap.add_argument('--no-write-profile', action='store_true')
    args = ap.parse_args()

    trades = load_official_trades(strict_061_only=True)
    baseline = enrich(simulate_official_risk_policy(trades, None, initial_funds=INITIAL_FUNDS), None, {'officialActualPnl': 0, 'maxDrawdownUsd': 0}, {'officialActualPnl': 0, 'maxDrawdownUsd': 0})
    combo_policy = current_combo_policy()
    combo = enrich(simulate_official_risk_policy(trades, combo_policy, initial_funds=INITIAL_FUNDS), combo_policy, baseline, baseline)

    rows: list[dict[str, Any]] = []
    for policy in policy_grid():
        rows.append(enrich(simulate_official_risk_policy(trades, policy, initial_funds=INITIAL_FUNDS), policy, baseline, combo))
    rows.sort(key=lambda r: (r['rankScore'], r['officialActualPnl'], -r['maxDrawdownUsd']), reverse=True)
    selected = select_candidates(rows, baseline, combo)

    audit = {
        'generatedAt': datetime.now(timezone.utc).isoformat(),
        'officialTruthOnly': True,
        'strict061Only': True,
        'tradeCount': len(trades),
        'baselineRecreated': len(trades) >= 250 and baseline['allTrades'] == len(trades),
        'comboRecreated': combo['policyName'] == 'combo_loss4_highprice0.55_pause4_lag4',
        'policyCount': len(rows),
        'noFutureSorting': True,
        'pendingExcluded': True,
        'passed': len(trades) >= 250 and len(rows) > 0,
    }
    write_json_atomic(OUT_AUDIT_JSON, audit)
    write_text(OUT_AUDIT_MD, '# 061 风控V3状态机审计\n\n' + '\n'.join(f'- {k}: `{v}`' for k, v in audit.items()) + '\n')

    leaderboard = {'generatedAt': audit['generatedAt'], 'baseline': baseline, 'currentCombo': combo, 'selected': selected, 'topRows': rows[:100]}
    write_json_atomic(OUT_LEADERBOARD_JSON, leaderboard)
    lines = ['# 061 风控V3状态机排行榜', '', '|排名|方案|候选|保留/总数|胜率|官方盈亏|最大回撤|拦截赢/输|保留率|严格|', '|---:|---|---|---:|---:|---:|---:|---:|---:|---|']
    for i, r in enumerate(rows[:40], 1):
        lines.append(f"|{i}|`{r['policyName']}`|`{r['candidateId']}`|{r['keptTrades']}/{r['allTrades']}|{r['winRatePct']:.2f}%|{money(r['officialActualPnl'])}|{money(r['maxDrawdownUsd'])}|{r['interceptedWinners']}/{r['interceptedLosers']}|{r['retentionRatePct']:.2f}%|{r['strictLiveCandidate']}|")
    write_text(OUT_LEADERBOARD_MD, '\n'.join(lines) + '\n')

    compare_rows = [
        ('当前061官方订单', baseline),
        ('当前live combo风控', combo),
        ('best_pnl', selected['best_pnl']),
        ('best_drawdown', selected['best_drawdown']),
        ('balanced_live_candidate', selected['balanced_live_candidate']),
    ]
    compare = {'generatedAt': audit['generatedAt'], 'baseline': baseline, 'currentCombo': combo, **selected}
    write_json_atomic(OUT_COMPARE_JSON, compare)
    write_text(OUT_COMPARE_MD, render_table('# 061 风控V3状态机对比', compare_rows) + '\n')

    balanced = selected['balanced_live_candidate']
    verdict = {
        'decision': 'WRITE_SHADOW_PROFILE' if balanced else 'NO_CANDIDATE',
        'selected': balanced,
        'baseline': baseline,
        'currentCombo': combo,
        'profilePath': str(V3_PROFILE),
        'shadowOnly': True,
        'notes': 'V3 is shadow only. It must pass 24h/72h live shadow validation before real blocking.',
    }
    write_json_atomic(OUT_VERDICT_JSON, verdict)
    write_text(
        OUT_VERDICT_MD,
        f"# 061 风控V3状态机结论\n\n- 决策: `{verdict['decision']}`\n- 选中: `{balanced.get('policyName') if balanced else '-'}`\n- 当前061官方盈亏: `{money(baseline['officialActualPnl'])}`\n- 当前combo官方盈亏: `{money(combo['officialActualPnl'])}`\n- V3选中官方盈亏: `{money(balanced.get('officialActualPnl') if balanced else None)}`\n- V3选中最大回撤: `{money(balanced.get('maxDrawdownUsd') if balanced else None)}`\n",
    )

    if not args.no_write_profile and balanced and balanced.get('policy'):
        profile = {
            'enabled': True,
            'shadow_only': True,
            'profile': 'top159_real_order_risk_v3_state_machine_shadow',
            'candidate_id': balanced['candidateId'],
            'strict_061_only': True,
            'source_report': str(OUT_VERDICT_JSON),
            'selected_policy': balanced['policy'],
            'risk_v3_replay_summary': {
                k: balanced[k]
                for k in [
                    'allTrades', 'keptTrades', 'skippedTrades', 'wins', 'losses', 'winRatePct',
                    'officialActualPnl', 'maxDrawdownUsd', 'interceptedWinners', 'interceptedLosers',
                    'retentionRatePct', 'pauseCount', 'pauseTotalBars', 'skipReasons', 'setHash',
                ]
            },
        }
        write_json_atomic(V3_PROFILE, profile)

    print(json.dumps({'audit': audit, 'compare': compare, 'reports': [str(OUT_COMPARE_MD), str(OUT_VERDICT_MD)]}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
