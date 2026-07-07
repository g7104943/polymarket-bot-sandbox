#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path('/Users/mac/polyfun')
NEXT = ROOT / 'polyfun-next'
REPORTS = ROOT / 'reports'
sys.path.insert(0, str(NEXT / 'src'))

from polyfun_next.risk_v2 import (  # noqa: E402
    V4_PROFILE,
    RiskV2Policy,
    load_official_trades,
    read_json,
    simulate_official_risk_policy,
    stable_hash,
    write_json_atomic,
)

INITIAL_FUNDS = 847.091209
OUT_AUDIT_JSON = REPORTS / 'top159_risk_v4_full_search_bug_audit_latest.json'
OUT_AUDIT_MD = REPORTS / 'top159_risk_v4_full_search_bug_audit_latest.md'
OUT_LEADERBOARD_JSON = REPORTS / 'top159_risk_v4_full_search_leaderboard_latest.json'
OUT_LEADERBOARD_MD = REPORTS / 'top159_risk_v4_full_search_leaderboard_latest.md'
OUT_COMPARE_JSON = REPORTS / 'top159_risk_v4_full_search_compare_latest.json'
OUT_COMPARE_MD = REPORTS / 'top159_risk_v4_full_search_compare_latest.md'
OUT_FAMILY_JSON = REPORTS / 'top159_risk_v4_full_search_family_best_latest.json'
OUT_VERDICT_JSON = REPORTS / 'top159_risk_v4_full_search_unique_verdict_latest.json'
OUT_VERDICT_MD = REPORTS / 'top159_risk_v4_full_search_unique_verdict_latest.md'


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


def current_v3_policy() -> RiskV2Policy | None:
    raw = read_json(NEXT / 'runtime' / 'top159_risk_profile_v3.json', {}) or {}
    pol = raw.get('selected_policy') if isinstance(raw.get('selected_policy'), dict) else None
    if not pol:
        return None
    allowed = set(RiskV2Policy.__dataclass_fields__)
    return RiskV2Policy(**{k: v for k, v in pol.items() if k in allowed})


def add(out: list[tuple[str, RiskV2Policy]], group: str, policy: RiskV2Policy) -> None:
    out.append((group, policy))


def old_policy_grid() -> list[tuple[str, RiskV2Policy]]:
    out: list[tuple[str, RiskV2Policy]] = []
    for lag in [1, 2, 4, 8]:
        for n in [3, 4, 5]:
            for bars in [4, 8, 12, 24, 48]:
                add(out, 'old_loss_streak', RiskV2Policy(name=f'loss_streak_n{n}_pause{bars}_lag{lag}', family='loss_streak', update_lag_bars=lag, loss_streak_n=n, loss_pause_bars=bars))
        for window in [5, 10, 20, 50]:
            for wr in [0.30, 0.35, 0.40, 0.45, 0.50]:
                for bars in [4, 8, 12, 24]:
                    add(out, 'old_rolling_winrate', RiskV2Policy(name=f'rolling_wr_w{window}_{wr}_pause{bars}_lag{lag}', family='rolling_winrate', update_lag_bars=lag, rolling_window=window, rolling_min_winrate=wr, rolling_pause_bars=bars))
            for pnl in [-10, -20, -30, -50, -75]:
                for bars in [4, 8, 12, 24]:
                    add(out, 'old_rolling_pnl', RiskV2Policy(name=f'rolling_pnl_w{window}_{pnl}_pause{bars}_lag{lag}', family='rolling_pnl', update_lag_bars=lag, rolling_window=window, rolling_min_pnl=float(pnl), rolling_pause_bars=bars))
        for dd in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]:
            for bars in [8, 16, 32, 96]:
                add(out, 'old_day_drawdown', RiskV2Policy(name=f'day_dd_{dd}_pause{bars}_lag{lag}', family='day_drawdown', update_lag_bars=lag, day_drawdown_fraction=dd, day_pause_bars=bars))
        for dd in [0.08, 0.10, 0.12, 0.15, 0.20]:
            for bars in [12, 24, 48, 96]:
                add(out, 'old_global_drawdown', RiskV2Policy(name=f'global_dd_{dd}_pause{bars}_lag{lag}', family='global_drawdown', update_lag_bars=lag, global_drawdown_fraction=dd, global_pause_bars=bars))
        for price in [0.55, 0.60, 0.65, 0.70]:
            for n in [2, 3, 4]:
                for bars in [4, 8, 12, 24, 48]:
                    add(out, 'old_high_price_loss', RiskV2Policy(name=f'high_price_{price}_loss{n}_pause{bars}_lag{lag}', family='high_price_loss', update_lag_bars=lag, high_price_min=price, high_price_loss_streak_n=n, high_price_pause_bars=bars))
        for score in [0.56, 0.57, 0.58, 0.60]:
            for n in [2, 3, 4]:
                for bars in [4, 8, 12, 24]:
                    add(out, 'old_low_score_loss', RiskV2Policy(name=f'low_score_{score}_loss{n}_pause{bars}_lag{lag}', family='low_score_loss', update_lag_bars=lag, low_score_max=score, low_score_loss_streak_n=n, low_score_pause_bars=bars))
        for n in [3, 4, 5]:
            for bars in [4, 8, 12, 24]:
                for window, pnl in [(10, -20), (20, -30), (50, -50)]:
                    add(out, 'old_combo_loss_rolling', RiskV2Policy(name=f'combo_loss{n}_roll{window}_{pnl}_pause{bars}_lag{lag}', family='combo_loss_rolling', update_lag_bars=lag, loss_streak_n=n, loss_pause_bars=bars, rolling_window=window, rolling_min_pnl=float(pnl), rolling_pause_bars=bars))
                for price in [0.55, 0.60, 0.65]:
                    add(out, 'old_combo_loss_highprice', RiskV2Policy(name=f'combo_loss{n}_highprice{price}_pause{bars}_lag{lag}', family='combo_loss_highprice', update_lag_bars=lag, loss_streak_n=n, loss_pause_bars=bars, high_price_min=price, high_price_loss_streak_n=2, high_price_pause_bars=bars))
    return out


def state_machine_grid() -> list[tuple[str, RiskV2Policy]]:
    out: list[tuple[str, RiskV2Policy]] = []
    for lag in [1, 2, 4, 8]:
        for yw in [10, 20, 30, 50]:
            for yp in [-8.0, -10.0, -15.0, -20.0, -30.0, -50.0]:
                for ybars in [4, 8, 12, 16]:
                    for rw in [10, 20, 30, 50]:
                        for rp in [-15.0, -20.0, -30.0, -40.0, -50.0, -75.0]:
                            for rbars in [8, 12, 24, 48]:
                                add(out, 'new_official_loss_state', RiskV2Policy(
                                    name=f'v4_lossstate_y{yw}_{int(yp)}p{ybars}_r{rw}_{int(rp)}p{rbars}_lag{lag}',
                                    family='state_machine', update_lag_bars=lag,
                                    state_yellow_window=yw, state_yellow_min_pnl=yp, state_yellow_pause_bars=ybars,
                                    state_red_window=rw, state_red_min_pnl=rp, state_red_pause_bars=rbars,
                                ))
        for hp in [0.52, 0.55, 0.58, 0.60, 0.65, 0.70]:
            for hp_loss_n in [2, 3, 4]:
                for bars in [4, 8, 12, 24, 48]:
                    add(out, 'new_price_loss_guard', RiskV2Policy(
                        name=f'v4_priceguard_hp{hp}_loss{hp_loss_n}_p{bars}_lag{lag}',
                        family='state_machine', update_lag_bars=lag,
                        state_red_high_price_min=hp, state_red_high_price_loss_streak_n=hp_loss_n, state_red_pause_bars=bars,
                    ))
        for price in [0.52, 0.55, 0.58, 0.60, 0.65]:
            for score in [0.55, 0.56, 0.57, 0.58, 0.60]:
                for n in [2, 3, 4]:
                    for bars in [4, 8, 12, 24]:
                        add(out, 'new_score_price_guard', RiskV2Policy(
                            name=f'v4_scoreprice_p{price}_s{score}_loss{n}_p{bars}_lag{lag}',
                            family='state_machine', update_lag_bars=lag,
                            state_red_score_price_min=price, state_red_score_price_max=score,
                            state_red_score_price_loss_streak_n=n, state_red_pause_bars=bars,
                        ))
        for ydd in [0.03, 0.05, 0.08, 0.10]:
            for gdd in [0.08, 0.10, 0.12, 0.15, 0.20]:
                for bars in [8, 12, 24, 48]:
                    add(out, 'new_drawdown_waterfall', RiskV2Policy(
                        name=f'v4_ddwater_day{ydd}_glob{gdd}_p{bars}_lag{lag}',
                        family='state_machine', update_lag_bars=lag,
                        state_red_day_drawdown_fraction=ydd, state_red_global_drawdown_fraction=gdd,
                        state_red_pause_bars=bars,
                    ))
        for rw, rp in [(10, -12.0), (20, -20.0), (20, -30.0), (30, -30.0), (50, -50.0)]:
            for hp in [0.55, 0.60, 0.65]:
                for loss_n in [3, 4, 5]:
                    for hp_loss_n in [2, 3]:
                        for bars in [8, 12, 24]:
                            add(out, 'new_hybrid_v4', RiskV2Policy(
                                name=f'v4_hybrid_r{rw}_{int(rp)}_loss{loss_n}_hp{hp}_hpl{hp_loss_n}_p{bars}_lag{lag}',
                                family='state_machine', update_lag_bars=lag,
                                state_yellow_window=10, state_yellow_min_pnl=-10.0, state_yellow_pause_bars=4,
                                state_red_window=rw, state_red_min_pnl=rp, state_red_loss_streak_n=loss_n,
                                state_red_high_price_min=hp, state_red_high_price_loss_streak_n=hp_loss_n,
                                state_red_pause_bars=bars,
                            ))
    return out


def all_policies() -> list[tuple[str, RiskV2Policy]]:
    raw = old_policy_grid() + state_machine_grid()
    seen: set[str] = set()
    out: list[tuple[str, RiskV2Policy]] = []
    for group, policy in raw:
        if policy.name in seen:
            continue
        seen.add(policy.name)
        out.append((group, policy))
    return out


def enrich(row: dict[str, Any], policy: RiskV2Policy | None, group: str, baseline: dict[str, Any], current_v3: dict[str, Any], train_row: dict[str, Any] | None, val_row: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(row)
    out['policy'] = asdict(policy) if policy else None
    out['familyGroup'] = group
    out['candidateId'] = stable_hash(out['policy']) if out['policy'] else 'baseline'
    out['pnlGainVs061'] = round(out['officialActualPnl'] - baseline['officialActualPnl'], 6)
    out['pnlGainVsCurrentV3'] = round(out['officialActualPnl'] - current_v3['officialActualPnl'], 6)
    out['drawdownReductionVs061'] = round(baseline['maxDrawdownUsd'] - out['maxDrawdownUsd'], 6)
    out['drawdownReductionVsCurrentV3'] = round(current_v3['maxDrawdownUsd'] - out['maxDrawdownUsd'], 6)
    out['loserMinusWinnerIntercepts'] = int(out['interceptedLosers']) - int(out['interceptedWinners'])
    if train_row:
        out['trainOfficialActualPnl'] = train_row['officialActualPnl']
        out['trainMaxDrawdownUsd'] = train_row['maxDrawdownUsd']
        out['trainRetentionRatePct'] = train_row['retentionRatePct']
    if val_row:
        out['validationOfficialActualPnl'] = val_row['officialActualPnl']
        out['validationMaxDrawdownUsd'] = val_row['maxDrawdownUsd']
        out['validationRetentionRatePct'] = val_row['retentionRatePct']
    validation_ok = True
    if val_row is not None:
        validation_ok = val_row['officialActualPnl'] >= 0 or val_row['officialActualPnl'] >= current_v3.get('validationOfficialActualPnl', -10**9)
    out['strictLiveCandidate'] = bool(
        out['officialActualPnl'] > current_v3['officialActualPnl']
        and out['officialActualPnl'] > baseline['officialActualPnl']
        and out['maxDrawdownUsd'] <= baseline['maxDrawdownUsd'] * 0.75
        and out['retentionRatePct'] >= 55.0
        and validation_ok
    )
    out['rankScore'] = round(
        out['pnlGainVsCurrentV3'] * 100.0
        + out['drawdownReductionVs061'] * 30.0
        + out['loserMinusWinnerIntercepts'] * 35.0
        + max(0.0, out['retentionRatePct'] - 55.0) * 3.0
        + (out.get('validationOfficialActualPnl', 0.0) * 30.0),
        6,
    )
    return out


def split_trades(trades: list[Any]) -> tuple[list[Any], list[Any]]:
    cut = max(1, min(len(trades) - 1, int(len(trades) * 0.60))) if len(trades) > 2 else len(trades)
    return trades[:cut], trades[cut:]


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
    ap = argparse.ArgumentParser(description='Full V4 official-order risk search for top159/061. Writes V4 shadow profile by default.')
    ap.add_argument('--no-write-profile', action='store_true')
    args = ap.parse_args()

    trades = load_official_trades(strict_061_only=True)
    train, val = split_trades(trades)
    baseline = enrich(simulate_official_risk_policy(trades, None, initial_funds=INITIAL_FUNDS), None, 'baseline', {'officialActualPnl': 0, 'maxDrawdownUsd': 0}, {'officialActualPnl': 0, 'maxDrawdownUsd': 0}, None, None)
    baseline_train = simulate_official_risk_policy(train, None, initial_funds=INITIAL_FUNDS)
    baseline_val = simulate_official_risk_policy(val, None, initial_funds=INITIAL_FUNDS) if val else None
    combo_policy = current_combo_policy()
    combo = enrich(simulate_official_risk_policy(trades, combo_policy, initial_funds=INITIAL_FUNDS), combo_policy, 'current_v2_combo', baseline, baseline, None, None)
    v3_policy = current_v3_policy()
    current_v3 = enrich(simulate_official_risk_policy(trades, v3_policy, initial_funds=INITIAL_FUNDS), v3_policy, 'current_v3_live', baseline, baseline, None, None) if v3_policy else combo
    current_v3['validationOfficialActualPnl'] = simulate_official_risk_policy(val, v3_policy, initial_funds=INITIAL_FUNDS)['officialActualPnl'] if (v3_policy and val) else 0.0

    rows: list[dict[str, Any]] = []
    for group, policy in all_policies():
        full = simulate_official_risk_policy(trades, policy, initial_funds=INITIAL_FUNDS)
        train_m = simulate_official_risk_policy(train, policy, initial_funds=INITIAL_FUNDS)
        val_m = simulate_official_risk_policy(val, policy, initial_funds=INITIAL_FUNDS) if val else None
        rows.append(enrich(full, policy, group, baseline, current_v3, train_m, val_m))
    rows.sort(key=lambda r: (r['rankScore'], r['officialActualPnl'], -r['maxDrawdownUsd'], r['retentionRatePct']), reverse=True)

    family_best: dict[str, dict[str, Any]] = {}
    for r in rows:
        g = r['familyGroup']
        if g not in family_best:
            family_best[g] = r
    strict = [r for r in rows if r.get('strictLiveCandidate')]
    pnl_pool = [r for r in rows if r['officialActualPnl'] > current_v3['officialActualPnl'] and r['retentionRatePct'] >= 55.0]
    best_pnl = max(pnl_pool or rows, key=lambda r: (r['officialActualPnl'], -r['maxDrawdownUsd'], r['retentionRatePct']))
    dd_pool = [r for r in rows if r['officialActualPnl'] >= current_v3['officialActualPnl'] - 10.0 and r['retentionRatePct'] >= 55.0]
    best_drawdown = min(dd_pool or rows, key=lambda r: (r['maxDrawdownUsd'], -r['officialActualPnl'], -r['retentionRatePct']))
    balanced = strict[0] if strict else (pnl_pool[0] if pnl_pool else rows[0])

    audit = {
        'generatedAt': datetime.now(timezone.utc).isoformat(),
        'officialTruthOnly': True,
        'strict061Only': True,
        'tradeCount': len(trades),
        'trainTrades': len(train),
        'validationTrades': len(val),
        'baselineRecreated': len(trades) >= 250 and baseline['allTrades'] == len(trades),
        'baselineOfficialActualPnl': baseline['officialActualPnl'],
        'currentV3OfficialActualPnl': current_v3['officialActualPnl'],
        'policyCount': len(rows),
        'familyCount': len(family_best),
        'strictPassCount': len(strict),
        'noFutureSorting': True,
        'pendingExcluded': True,
        'splitValidationIncluded': bool(val),
        'passed': len(trades) >= 250 and len(rows) > 0,
    }
    write_json_atomic(OUT_AUDIT_JSON, audit)
    write_text(OUT_AUDIT_MD, '# 061 风控 V4 全量搜索审计\n\n' + '\n'.join(f'- {k}: `{v}`' for k, v in audit.items()) + '\n')

    leaderboard = {
        'generatedAt': audit['generatedAt'],
        'baseline': baseline,
        'baselineTrain': baseline_train,
        'baselineValidation': baseline_val,
        'currentV2Combo': combo,
        'currentV3Live': current_v3,
        'selected': {'best_pnl': best_pnl, 'best_drawdown': best_drawdown, 'balanced_live_candidate': balanced},
        'familyBest': family_best,
        'topRows': rows[:200],
    }
    write_json_atomic(OUT_LEADERBOARD_JSON, leaderboard)
    write_json_atomic(OUT_FAMILY_JSON, family_best)

    lines = ['# 061 风控 V4 全量搜索排行榜', '', '|排名|规则族|方案|候选|保留/总数|胜率|官方盈亏|最大回撤|拦截赢/输|验证盈亏|保留率|严格|', '|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for i, r in enumerate(rows[:60], 1):
        lines.append(f"|{i}|{r['familyGroup']}|`{r['policyName']}`|`{r['candidateId']}`|{r['keptTrades']}/{r['allTrades']}|{r['winRatePct']:.2f}%|{money(r['officialActualPnl'])}|{money(r['maxDrawdownUsd'])}|{r['interceptedWinners']}/{r['interceptedLosers']}|{money(r.get('validationOfficialActualPnl'))}|{r['retentionRatePct']:.2f}%|{r['strictLiveCandidate']}|")
    write_text(OUT_LEADERBOARD_MD, '\n'.join(lines) + '\n')

    compare_rows = [
        ('裸061官方订单', baseline),
        ('当前V2 combo', combo),
        ('当前V3 live', current_v3),
        ('V4 best_pnl', best_pnl),
        ('V4 best_drawdown', best_drawdown),
        ('V4 balanced', balanced),
    ]
    compare = {'generatedAt': audit['generatedAt'], 'baseline': baseline, 'currentV2Combo': combo, 'currentV3Live': current_v3, 'best_pnl': best_pnl, 'best_drawdown': best_drawdown, 'balanced_live_candidate': balanced}
    write_json_atomic(OUT_COMPARE_JSON, compare)
    write_text(OUT_COMPARE_MD, render_table('# 061 风控 V4 全量搜索对比', compare_rows) + '\n')

    has_strict = bool(strict)
    verdict = {
        'decision': 'WRITE_V4_SHADOW_PROFILE' if has_strict and balanced else 'NO_STRICT_PROFILE_KEEP_CURRENT_V3',
        'selected': balanced,
        'baseline': baseline,
        'currentV3Live': current_v3,
        'profilePath': str(V4_PROFILE),
        'shadowOnly': True,
        'notes': 'V4 is shadow-only. Current live V3 remains unchanged.',
    }
    write_json_atomic(OUT_VERDICT_JSON, verdict)
    write_text(
        OUT_VERDICT_MD,
        f"# 061 风控 V4 全量搜索结论\n\n- 决策: `{verdict['decision']}`\n- 选中: `{balanced.get('policyName') if balanced else '-'}`\n- 当前V3官方盈亏: `{money(current_v3['officialActualPnl'])}`\n- V4选中官方盈亏: `{money(balanced.get('officialActualPnl') if balanced else None)}`\n- V4选中最大回撤: `{money(balanced.get('maxDrawdownUsd') if balanced else None)}`\n- V4选中保留率: `{balanced.get('retentionRatePct') if balanced else '-'}%`\n",
    )

    if not args.no_write_profile and balanced and balanced.get('policy'):
        profile = {
            'enabled': bool(has_strict),
            'shadow_only': True,
            'profile': 'top159_real_order_risk_v4_full_search_shadow' if has_strict else 'top159_real_order_risk_v4_observation_disabled',
            'candidate_id': balanced['candidateId'],
            'strict_061_only': True,
            'source_report': str(OUT_VERDICT_JSON),
            'selected_policy': balanced['policy'],
            'no_strict_pass_reason': None if has_strict else 'No V4 policy beat current V3 under strict criteria; kept disabled.',
            'risk_v4_replay_summary': {k: balanced[k] for k in [
                'allTrades', 'keptTrades', 'skippedTrades', 'wins', 'losses', 'winRatePct',
                'officialActualPnl', 'maxDrawdownUsd', 'interceptedWinners', 'interceptedLosers',
                'retentionRatePct', 'pauseCount', 'pauseTotalBars', 'skipReasons', 'setHash',
                'familyGroup', 'validationOfficialActualPnl',
            ] if k in balanced},
        }
        write_json_atomic(V4_PROFILE, profile)

    print(json.dumps({'audit': audit, 'selected': balanced, 'reports': [str(OUT_COMPARE_MD), str(OUT_VERDICT_MD), str(OUT_LEADERBOARD_MD)]}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
