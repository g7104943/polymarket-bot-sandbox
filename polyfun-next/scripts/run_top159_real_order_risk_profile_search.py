#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path('/Users/mac/polyfun')
NEXT = ROOT / 'polyfun-next'
REPORTS = ROOT / 'reports'
RUNTIME = NEXT / 'runtime'
import sys
sys.path.insert(0, str(NEXT / 'src'))

from polyfun_next.risk_v2 import (  # noqa: E402
    DEFAULT_PROFILE,
    RiskV2Policy,
    load_official_trades,
    simulate_official_risk_policy,
    stable_hash,
    write_json_atomic,
)

OUT_AUDIT_JSON = REPORTS / 'top159_real_order_risk_v2_bug_audit_latest.json'
OUT_AUDIT_MD = REPORTS / 'top159_real_order_risk_v2_bug_audit_latest.md'
OUT_LEADERBOARD_JSON = REPORTS / 'top159_real_order_risk_v2_leaderboard_latest.json'
OUT_LEADERBOARD_MD = REPORTS / 'top159_real_order_risk_v2_leaderboard_latest.md'
OUT_COMPARE_JSON = REPORTS / 'top159_real_order_risk_v2_compare_latest.json'
OUT_COMPARE_MD = REPORTS / 'top159_real_order_risk_v2_compare_latest.md'
OUT_VERDICT_JSON = REPORTS / 'top159_real_order_risk_v2_unique_verdict_latest.json'
OUT_VERDICT_MD = REPORTS / 'top159_real_order_risk_v2_unique_verdict_latest.md'

INITIAL_FUNDS = 847.091209


def money(x: float | int | None) -> str:
    if x is None:
        return '-'
    return f'{float(x):+,.2f}'


def pct(x: float | int | None) -> str:
    if x is None:
        return '-'
    return f'{float(x):.2f}%'


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def policy_grid() -> list[RiskV2Policy]:
    out: list[RiskV2Policy] = []
    for lag in [1, 2, 4, 8]:
        for n in [3, 4, 5]:
            for bars in [4, 8, 12, 24, 48]:
                out.append(RiskV2Policy(name=f'loss_streak_n{n}_pause{bars}_lag{lag}', family='loss_streak', update_lag_bars=lag, loss_streak_n=n, loss_pause_bars=bars))
        for window in [5, 10, 20, 50]:
            for wr in [0.30, 0.35, 0.40, 0.45, 0.50]:
                for bars in [4, 8, 12, 24]:
                    out.append(RiskV2Policy(name=f'rolling_wr_w{window}_{wr}_pause{bars}_lag{lag}', family='rolling_winrate', update_lag_bars=lag, rolling_window=window, rolling_min_winrate=wr, rolling_pause_bars=bars))
            for pnl in [-10, -20, -30, -50, -75]:
                for bars in [4, 8, 12, 24]:
                    out.append(RiskV2Policy(name=f'rolling_pnl_w{window}_{pnl}_pause{bars}_lag{lag}', family='rolling_pnl', update_lag_bars=lag, rolling_window=window, rolling_min_pnl=float(pnl), rolling_pause_bars=bars))
        for dd in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]:
            for bars in [8, 16, 32, 96]:
                out.append(RiskV2Policy(name=f'day_dd_{dd}_pause{bars}_lag{lag}', family='day_drawdown', update_lag_bars=lag, day_drawdown_fraction=dd, day_pause_bars=bars))
        for dd in [0.08, 0.10, 0.12, 0.15, 0.20]:
            for bars in [12, 24, 48, 96]:
                out.append(RiskV2Policy(name=f'global_dd_{dd}_pause{bars}_lag{lag}', family='global_drawdown', update_lag_bars=lag, global_drawdown_fraction=dd, global_pause_bars=bars))
        for price in [0.55, 0.60, 0.65, 0.70]:
            for n in [2, 3, 4]:
                for bars in [4, 8, 12, 24, 48]:
                    out.append(RiskV2Policy(name=f'high_price_{price}_loss{n}_pause{bars}_lag{lag}', family='high_price_loss', update_lag_bars=lag, high_price_min=price, high_price_loss_streak_n=n, high_price_pause_bars=bars))
        for score in [0.56, 0.57, 0.58, 0.60]:
            for n in [2, 3, 4]:
                for bars in [4, 8, 12, 24]:
                    out.append(RiskV2Policy(name=f'low_score_{score}_loss{n}_pause{bars}_lag{lag}', family='low_score_loss', update_lag_bars=lag, low_score_max=score, low_score_loss_streak_n=n, low_score_pause_bars=bars))
        # Compact combination family: the exact user problem (loss streak) plus
        # either rolling PnL or high-price loss protection.
        for n in [3, 4, 5]:
            for bars in [4, 8, 12, 24]:
                for window, pnl in [(10, -20), (20, -30), (50, -50)]:
                    out.append(RiskV2Policy(name=f'combo_loss{n}_roll{window}_{pnl}_pause{bars}_lag{lag}', family='combo_loss_rolling', update_lag_bars=lag, loss_streak_n=n, loss_pause_bars=bars, rolling_window=window, rolling_min_pnl=float(pnl), rolling_pause_bars=bars))
                for price in [0.55, 0.60, 0.65]:
                    out.append(RiskV2Policy(name=f'combo_loss{n}_highprice{price}_pause{bars}_lag{lag}', family='combo_loss_highprice', update_lag_bars=lag, loss_streak_n=n, loss_pause_bars=bars, high_price_min=price, high_price_loss_streak_n=2, high_price_pause_bars=bars))
    # De-duplicate names just in case.
    seen: set[str] = set()
    uniq: list[RiskV2Policy] = []
    for p in out:
        if p.name in seen:
            continue
        seen.add(p.name)
        uniq.append(p)
    return uniq


def score_row(row: dict[str, Any], baseline: dict[str, Any]) -> float:
    pnl_gain = row['officialActualPnl'] - baseline['officialActualPnl']
    dd_gain = baseline['maxDrawdownUsd'] - row['maxDrawdownUsd']
    save_loss = row['interceptedLosers'] - row['interceptedWinners']
    retention_penalty = max(0.0, 50.0 - row['retentionRatePct']) * 8.0
    # PnL is first priority, drawdown second, then whether the rule truly
    # blocks more losers than winners. Avoid over-rewarding tiny retention.
    return pnl_gain * 100.0 + dd_gain * 15.0 + save_loss * 25.0 - retention_penalty


def fail_reasons(row: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if row['retentionRatePct'] < 50.0:
        reasons.append('retention_below_50pct')
    if row['officialActualPnl'] <= baseline['officialActualPnl']:
        reasons.append('pnl_not_better_than_actual_061')
    if row['maxDrawdownUsd'] > baseline['maxDrawdownUsd'] and row['officialActualPnl'] < baseline['officialActualPnl'] + 50.0:
        reasons.append('drawdown_worse_without_large_pnl_gain')
    if row['interceptedLosers'] < row['interceptedWinners'] and row['officialActualPnl'] < 0:
        reasons.append('blocks_more_winners_than_losers_while_still_losing')
    return reasons


def run_search(strict_061_only: bool) -> dict[str, Any]:
    trades = load_official_trades(strict_061_only=strict_061_only)
    baseline = simulate_official_risk_policy(trades, None, initial_funds=INITIAL_FUNDS)
    rows: list[dict[str, Any]] = []
    for p in policy_grid():
        m = simulate_official_risk_policy(trades, p, initial_funds=INITIAL_FUNDS)
        reasons = fail_reasons(m, baseline)
        m['policy'] = asdict(p)
        m['strictPass'] = not reasons
        m['failReasons'] = reasons
        m['rankScore'] = score_row(m, baseline)
        rows.append(m)
    rows.sort(key=lambda r: r['rankScore'], reverse=True)
    strict = [r for r in rows if r['strictPass']]
    return {
        'strict061Only': strict_061_only,
        'generatedAt': datetime.now(timezone.utc).isoformat(),
        'initialFunds': INITIAL_FUNDS,
        'baseline': baseline,
        'bestOverall': rows[0] if rows else None,
        'bestStrictPass': strict[0] if strict else None,
        'strictPassCount': len(strict),
        'topRows': rows[:100],
    }


def write_reports(result: dict[str, Any], *, apply_profile: bool) -> None:
    baseline = result['baseline']
    best = result.get('bestStrictPass') or result.get('bestOverall')
    strict_rows = [r for r in result['topRows'] if r.get('strictPass')]
    loss_aware = [
        r for r in strict_rows
        if int(((r.get('policy') or {}).get('loss_streak_n') or 0)) > 0
        and r.get('officialActualPnl', -10**9) >= (best or {}).get('officialActualPnl', -10**9) - 1.0
        and r.get('maxDrawdownUsd', 10**9) <= (best or {}).get('maxDrawdownUsd', 10**9) + 1.0
    ]
    recommended = loss_aware[0] if loss_aware else best
    audit = {
        'researchOnlyNoLiveOrderChange': True,
        'officialTruthOnly': True,
        'strict061Only': result['strict061Only'],
        'baselineTrades': baseline['allTrades'],
        'baselineOfficialPnl': baseline['officialActualPnl'],
        'baselineMaxDrawdown': baseline['maxDrawdownUsd'],
        'baselineRecreated': baseline['allTrades'] >= 250,
        'policyCount': len(policy_grid()),
        'passed': baseline['allTrades'] >= 250 and best is not None,
    }
    write_json_atomic(OUT_AUDIT_JSON, audit)
    write_text(OUT_AUDIT_MD, '# 061真实订单风控V2审计\n\n' + '\n'.join(f'- {k}: `{v}`' for k, v in audit.items()) + '\n')
    write_json_atomic(OUT_LEADERBOARD_JSON, result['topRows'])
    lines = [
        '# 061真实订单风控V2排行榜',
        '',
        '|排名|方案|通过|保留/总数|胜率|官方盈亏|最大回撤|拦截赢/输|暂停次数|最长连亏|原因|',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for i, r in enumerate(result['topRows'][:30], 1):
        lines.append(
            f"|{i}|`{r['policyName']}`|{r['strictPass']}|{r['keptTrades']}/{r['allTrades']}|{r['winRatePct']:.2f}%|"
            f"{money(r['officialActualPnl'])}|{money(r['maxDrawdownUsd'])}|{r['interceptedWinners']}/{r['interceptedLosers']}|"
            f"{r['pauseCount']}|{r['longestSettledLossStreak']}|{','.join(r['failReasons'])}|"
        )
    write_text(OUT_LEADERBOARD_MD, '\n'.join(lines) + '\n')
    compare = {
        'baseline061OfficialActual': baseline,
        'bestOverall': result.get('bestOverall'),
        'bestStrictPass': result.get('bestStrictPass'),
        'recommendedShadowProfile': recommended,
    }
    write_json_atomic(OUT_COMPARE_JSON, compare)
    cm = ['# 061真实订单风控V2对比', '', '|配置|保留/总数|胜/负|胜率|官方盈亏|最大回撤|拦截赢/输|均价|胜单均价|输单均价|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, r in [('当前061真实官方订单', baseline), ('最佳总分风控', result.get('bestOverall')), ('最佳合格风控', result.get('bestStrictPass')), ('推荐影子风控', recommended)]:
        if not r:
            continue
        cm.append(
            f"|{name}|{r['keptTrades']}/{r['allTrades']}|{r['wins']}/{r['losses']}|{r['winRatePct']:.2f}%|"
            f"{money(r['officialActualPnl'])}|{money(r['maxDrawdownUsd'])}|{r['interceptedWinners']}/{r['interceptedLosers']}|"
            f"{r['weightedAvgBuyPrice']:.4f}|{r['winnerAvgBuyPrice']:.4f}|{r['loserAvgBuyPrice']:.4f}|"
        )
    write_text(OUT_COMPARE_MD, '\n'.join(cm) + '\n')
    verdict = {
        'decision': 'FOUND_RISK_PROFILE_FOR_SHADOW' if result.get('bestStrictPass') else 'NO_STRICT_PROFILE; use best overall only as observation',
        'selected': recommended,
        'baseline': baseline,
        'profilePath': str(DEFAULT_PROFILE),
        'applyProfile': apply_profile,
    }
    write_json_atomic(OUT_VERDICT_JSON, verdict)
    write_text(OUT_VERDICT_MD, f"# 061真实订单风控V2结论\n\n- 决策: `{verdict['decision']}`\n- 选中: `{(verdict['selected'] or {}).get('policyName')}`\n- 当前061官方盈亏: `{money(baseline['officialActualPnl'])}`\n- 选中官方盈亏: `{money((verdict['selected'] or {}).get('officialActualPnl'))}`\n")
    if apply_profile and recommended:
        profile = {
            'enabled': True,
            'shadow_only': True,
            'profile': 'top159_real_order_risk_v2_shadow',
            'candidate_id': stable_hash(recommended['policy']),
            'strict_061_only': result['strict061Only'],
            'source_report': str(OUT_VERDICT_JSON),
            'selected_policy': recommended['policy'],
            'selected_metrics': {k: recommended[k] for k in ['allTrades','keptTrades','skippedTrades','officialActualPnl','maxDrawdownUsd','interceptedWinners','interceptedLosers','retentionRatePct']},
        }
        write_json_atomic(DEFAULT_PROFILE, profile)


def main() -> int:
    ap = argparse.ArgumentParser(description='Search risk profiles on official top159/061 actual orders. Research only; optional shadow profile write.')
    ap.add_argument('--all-top159', action='store_true', help='Use all top159 official rows instead of strict 061 rows')
    ap.add_argument('--apply-shadow-profile', action='store_true', help='Write best profile to runtime/top159_risk_profile_v2.json with shadow_only=true')
    args = ap.parse_args()
    result = run_search(strict_061_only=not args.all_top159)
    write_reports(result, apply_profile=args.apply_shadow_profile)
    print(json.dumps({
        'baseline': result['baseline'],
        'bestOverall': result.get('bestOverall'),
        'bestStrictPass': result.get('bestStrictPass'),
        'reports': [str(OUT_COMPARE_MD), str(OUT_VERDICT_MD)],
    }, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
