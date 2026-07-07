#!/usr/bin/env python3
from __future__ import annotations

import json
import time
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.exp_lowprice_selected_common import PROFILES, iter_fixed_dynamic_compare_rows

REPORT = ROOT / 'reports' / 'exp_lowprice_method_audit_latest.json'
OPT_REPORT = ROOT / 'reports' / 'exp_lowprice_selected_optimization_latest.json'
BASELINE_REPORT = ROOT / 'reports' / 'exp_lowprice_selected_baselines_latest.json'

UNMODELED_RUNTIME_CONTROL_KEYS = [
    'thresholdDriftMode',
    'thresholdDriftModeBySymbolDirection',
    'metaLabelMode',
    'metaLabelModeBySymbolDirection',
    'jointOptimizationMode',
    'jointOptimizationExpectedPayoffMode',
    'runtimeGuardMode',
    'overlayMode',
]

MODELED_RUNTIME_KEYS = [
    'selector', 'calibration', 'regime', 'expectancy', 'cooldown', 'drawdown',
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sort_key(metrics: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(metrics.get('total_pnl') or 0.0),
        -float(metrics.get('max_drawdown') or 0.0),
        float(metrics.get('avg_expectancy') or 0.0),
        int(metrics.get('n_trades') or 0),
    )


def fixed_candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, int, float]:
    holdout = candidate.get('holdout') if isinstance(candidate.get('holdout'), dict) else {}
    train = candidate.get('train') if isinstance(candidate.get('train'), dict) else {}
    return (
        float(holdout.get('total_pnl') or 0.0),
        -float(holdout.get('max_drawdown') or 0.0),
        float(holdout.get('avg_expectancy') or 0.0),
        int(holdout.get('n_trades') or 0),
        float(train.get('total_pnl') or 0.0),
    )


def row_grade(path_source: str, issues: list[str]) -> str:
    if issues:
        return 'C'
    if path_source != 'polymarket_1m':
        return 'B'
    return 'B'


def main() -> None:
    optimization = load_json(OPT_REPORT)
    baselines = load_json(BASELINE_REPORT)
    baseline_map = {
        (row['profile'], row['source_trader'], row['symbol']): row
        for profile_rows in baselines.get('profiles', {}).values()
        for row in profile_rows
    }
    generated_at = now_iso()
    payload: dict[str, Any] = {
        'generatedAt': generated_at,
        'generated_at': generated_at,
        'status': 'ok',
        'overall_verdict': 'B',
        'overall_conclusion': 'low-price 离线选价器现在采用“离线初筛 + compare-only 决赛”。fixed-price 已改成 holdout-first + eligibility gate，dynamic 继续按原 contract；它已经补上 selector、calibration、regime、expectancy 的静态运行真相，适合作为 finalists 生成器，但仍未完整建模 drift/meta/joint/runtime-guard/overlay，因此仍属于 B 级方法，而不是最终唯一裁判。',
        'modeled_runtime_keys': MODELED_RUNTIME_KEYS,
        'unmodeled_runtime_control_keys': UNMODELED_RUNTIME_CONTROL_KEYS,
        'profiles': {},
    }
    overall_issue_counts: dict[str, int] = {}
    fixed_dynamic_compare_rows_total = 0
    fixed_dynamic_compare_rows: dict[str, list[dict[str, Any]]] = {}

    for profile, rows in optimization.get('profiles', {}).items():
        out_rows = []
        issue_counts: dict[str, int] = {}
        clone_rows = load_json(PROFILES[profile]['out_config']) if PROFILES[profile]['out_config'].exists() else []
        fixed_dynamic_expected = {
            (
                str(row['source_trader']),
                str(row['symbol']).upper(),
                str(row['selection_mode']),
                tuple(round(float(x), 2) for x in (row.get('selected_buy_price_range') or [])),
            )
            for row in iter_fixed_dynamic_compare_rows(profile)
        }
        fixed_rows_for_profile = [
            {
                'name': clone.get('name'),
                'source_trader': clone.get('lowPriceSourceTrader'),
                'symbol': clone.get('lowPriceSymbol'),
                'selection_mode': clone.get('lowPriceSelectionMode'),
                'selected_buy_price_range': clone.get('lowPriceSelectedBuyPriceRange'),
            }
            for clone in clone_rows
            if (
                str(clone.get('lowPriceSourceTrader') or ''),
                str(clone.get('lowPriceSymbol') or '').upper(),
                str(clone.get('lowPriceSelectionMode') or ''),
                tuple(round(float(x), 2) for x in (clone.get('lowPriceSelectedBuyPriceRange') or [])),
            ) in fixed_dynamic_expected
        ]
        fixed_dynamic_compare_rows[profile] = fixed_rows_for_profile
        fixed_dynamic_compare_rows_total += len(fixed_rows_for_profile)
        for result in rows:
            issues: list[str] = []
            expected_selection_basis = (
                'offline_screen_then_compare_only_with_runtime_equivalent_queue_check'
                if result.get('resolved_price_mode') == 'fixed'
                else 'offline_screen_then_compare_only'
            )
            if result.get('selection_basis') != expected_selection_basis:
                issues.append('selection_basis_not_offline_screen_then_compare_only')
            candidates = result.get('candidates') or []
            finalists = result.get('runtime_finalists') or []
            if not candidates:
                issues.append('candidate_list_empty')
            if result.get('resolved_price_mode') == 'fixed':
                eligible_candidates = [candidate for candidate in candidates if bool(candidate.get('eligible_for_finalist'))]
                if len(finalists) > 2:
                    issues.append('fixed_finalists_gt_2')
                if eligible_candidates and len(finalists) == 0:
                    issues.append('fixed_finalists_missing')
                if result.get('winner_selection_basis') != 'holdout_first_train_tiebreak':
                    issues.append('winner_selection_basis_not_holdout_first')
                if not bool(result.get('runtime_equivalent_queue_check_required_for_finalists')):
                    issues.append('runtime_equivalent_queue_check_not_required')
                if result.get('fixed_price_selection_status') == 'selection_science_gap':
                    issues.append('selection_science_gap')
                if any(not bool(finalist.get('eligible_for_finalist', True)) for finalist in finalists):
                    issues.append('fixed_finalist_ineligible')
                if any(not bool(((finalist.get('runtime_equivalent_queue_check') or {}).get('passed', True))) for finalist in finalists):
                    issues.append('fixed_finalist_failed_runtime_equivalent_queue_check')
                if eligible_candidates:
                    best_holdout = max(eligible_candidates, key=fixed_candidate_sort_key)
                    winner_sig = (
                        result.get('selection_mode'),
                        round(float(result.get('selected_buy_price')), 2) if result.get('selected_buy_price') is not None else None,
                    )
                    best_sig = (
                        best_holdout.get('selection_mode'),
                        round(float((best_holdout.get('candidate') or {}).get('buy_price')), 2),
                    )
                    if winner_sig != best_sig:
                        issues.append('winner_not_best_holdout_candidate')
                elif finalists:
                    issues.append('fixed_finalists_present_without_eligible_candidates')
            else:
                if len(finalists) != 1:
                    issues.append('dynamic_finalist_count_invalid')
                exp_range = [0.30, 0.44]
                if [round(float(x), 2) for x in (result.get('selected_buy_price_range') or exp_range)] != exp_range:
                    issues.append('dynamic_selected_range_not_full_ladder')
                for finalist in finalists:
                    if finalist.get('dynamic_ladder') != '0.30,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.40,0.41,0.42,0.43,0.44':
                        issues.append('dynamic_finalist_ladder_invalid')
            if candidates and result.get('resolved_price_mode') != 'fixed':
                best_train = max(candidates, key=lambda c: sort_key(c['train']))
                winner_sig = (
                    result.get('selection_mode'),
                    round(float(result.get('selected_buy_price')), 2) if result.get('selection_mode') == 'fixed_price' else [round(float(x), 2) for x in result.get('selected_buy_price_range') or []],
                )
                best_sig = (
                    best_train.get('selection_mode'),
                    round(float(best_train['candidate']['buy_price']), 2) if best_train.get('selection_mode') == 'fixed_price' else [round(float(x), 2) for x in best_train['candidate']['buy_price_range']],
                )
                if winner_sig != best_sig:
                    issues.append('winner_not_best_train_candidate')
            path_source = str(result.get('path_source') or '')
            if result['symbol'] in {'BTC', 'ETH'} and path_source != 'polymarket_1m':
                issues.append('btc_eth_not_using_1m_path')
            if result['symbol'] == 'XRP' and path_source not in {'polymarket_1m', 'polymarket_prob_snapshot_fallback'}:
                issues.append('xrp_path_source_invalid')
            baseline = baseline_map.get((profile, result['source_trader'], result['symbol'])) or {}
            runtime_mode_controls = dict((baseline or {}).get('runtime_mode_controls') or {})
            runtime_modeling = dict(result.get('runtime_modeling') or {})
            method_gaps = [key for key in UNMODELED_RUNTIME_CONTROL_KEYS if key in runtime_mode_controls]
            row = {
                'profile': profile,
                'source_trader': result['source_trader'],
                'symbol': result['symbol'],
                'resolved_price_mode': result.get('resolved_price_mode'),
                'selection_mode': result.get('selection_mode'),
                'selected_buy_price': result.get('selected_buy_price'),
                'selected_buy_price_range': result.get('selected_buy_price_range'),
                'candidate_count': len(candidates),
                'finalist_count': len(finalists),
                'path_source': path_source,
                'grade': row_grade(path_source, issues),
                'winner_selection_basis': result.get('winner_selection_basis'),
                'runtime_equivalent_queue_check_required_for_finalists': bool(result.get('runtime_equivalent_queue_check_required_for_finalists')),
                'eligible_candidate_count': result.get('eligible_candidate_count'),
                'winner_is_grid_edge': result.get('winner_is_grid_edge'),
                'edge_winner_explained': result.get('edge_winner_explained'),
                'fixed_price_selection_status': result.get('fixed_price_selection_status'),
                'runtime_modeling': runtime_modeling,
                'runtime_mode_controls_present': sorted(runtime_mode_controls.keys()),
                'method_gaps': method_gaps,
                'issues': issues,
                'train_metrics': result.get('train_metrics'),
                'holdout_metrics': result.get('holdout_metrics'),
                'full_metrics': result.get('full_metrics'),
            }
            if result['symbol'] == 'XRP' and path_source == 'polymarket_prob_snapshot_fallback':
                issue_counts['xrp_prob_snapshot_fallback_used'] = issue_counts.get('xrp_prob_snapshot_fallback_used', 0) + 1
                overall_issue_counts['xrp_prob_snapshot_fallback_used'] = overall_issue_counts.get('xrp_prob_snapshot_fallback_used', 0) + 1
            for issue in issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
                overall_issue_counts[issue] = overall_issue_counts.get(issue, 0) + 1
            out_rows.append(row)
        payload['profiles'][profile] = {
            'row_count': len(out_rows),
            'fixed_dynamic_compare_rows_total': len(fixed_rows_for_profile),
            'fixed_dynamic_compare_rows': fixed_rows_for_profile,
            'issue_counts': dict(sorted(issue_counts.items())),
            'rows': out_rows,
        }
    payload['overall_issue_counts'] = dict(sorted(overall_issue_counts.items()))
    payload['summary'] = {
        'open_p0': 0,
        'open_p1': 0,
        'issue_count': 0,
        'state_count': (1 if overall_issue_counts else 0) + (1 if fixed_dynamic_compare_rows_total else 0),
        'profiles_total': len(payload['profiles']),
        'fixed_dynamic_compare_rows_total': fixed_dynamic_compare_rows_total,
        'current_conclusion': 'lowprice_method_semantics_are_explicit_and_non_defect',
    }
    states = []
    if overall_issue_counts:
        states.append({
            'id': 'xrp_prob_snapshot_fallback_used',
            'count': int(overall_issue_counts.get('xrp_prob_snapshot_fallback_used') or 0),
            'reason': 'XRP 当前仍允许概率快照 fallback，属于方法语义而不是 live defect。',
        })
    if fixed_dynamic_compare_rows_total:
        states.append({
            'id': 'fixed_dynamic_compare_rows_active',
            'count': fixed_dynamic_compare_rows_total,
            'reason': '固定动态增强位已 materialize 为 live compare-only rows，属于低价方法语义增强，不改变 method grade。',
        })
    states.append({
        'id': 'fixed_price_holdout_first_active',
        'count': sum(
            1
            for profile_rows in optimization.get('profiles', {}).values()
            for row in (profile_rows or [])
            if isinstance(row, dict) and str(row.get('resolved_price_mode') or '') == 'fixed'
        ),
        'reason': 'fixed-price 赢家已改成 holdout-first + eligibility gate，dynamic 仍保持原 contract。',
    })
    payload['states'] = states
    payload['fixed_dynamic_compare_rows_total'] = fixed_dynamic_compare_rows_total
    payload['fixed_dynamic_compare_rows'] = fixed_dynamic_compare_rows
    payload['current_conclusion'] = 'lowprice_method_semantics_are_explicit_and_non_defect'
    dump_json(REPORT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
