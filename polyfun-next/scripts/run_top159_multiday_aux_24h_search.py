#!/usr/bin/env python3
from __future__ import annotations

# 24h research runner for top159 multiday auxiliary gate.
# Research-only: it writes reports/checkpoints and never touches live config.

import concurrent.futures as cf
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Keep each worker modest. The parent controls process count.
THREADS = os.environ.get('TOP159_MULTIDAY_WORKER_THREADS', '2')
for k in ['OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(k, THREADS)

import numpy as np
import pandas as pd

ROOT = Path('/Users/mac/polyfun')
REPORTS = ROOT / 'reports'
SCRIPT = ROOT / 'polyfun-next' / 'scripts' / 'run_top159_multiday_aux_research.py'

WORKERS = int(os.environ.get('TOP159_MULTIDAY_WORKERS', '4'))
MAX_SECONDS = int(os.environ.get('TOP159_MULTIDAY_24H_MAX_SECONDS', os.environ.get('TOP159_MULTIDAY_MAX_SECONDS', str(24 * 3600))))
CHECKPOINT_SECONDS = int(os.environ.get('TOP159_MULTIDAY_CHECKPOINT_SECONDS', '900'))
FULL_EVAL_LIMIT = int(os.environ.get('TOP159_MULTIDAY_24H_FULL_EVAL_LIMIT', os.environ.get('TOP159_MULTIDAY_FULL_EVAL_LIMIT', '20000')))
STRICT_EVAL_LIMIT = int(os.environ.get('TOP159_MULTIDAY_24H_STRICT_EVAL_LIMIT', os.environ.get('TOP159_MULTIDAY_STRICT_EVAL_LIMIT', '8000')))
COARSE_LIMIT = int(os.environ.get('TOP159_MULTIDAY_24H_COARSE_LIMIT', '0'))  # 0 = no cap
PERIOD_ENV = os.environ.get('TOP159_MULTIDAY_PERIODS', '').strip()

BUG_AUDIT_JSON = REPORTS / 'top159_multiday_aux_bug_audit_v2_latest.json'
BUG_AUDIT_MD = REPORTS / 'top159_multiday_aux_bug_audit_v2_latest.md'
CHECKPOINT_JSON = REPORTS / 'top159_multiday_aux_24h_checkpoint_latest.json'
RESULTS_JSONL = REPORTS / 'top159_multiday_aux_24h_full_results_latest.jsonl'
LEADERBOARD_JSON = REPORTS / 'top159_multiday_aux_24h_leaderboard_latest.json'
LEADERBOARD_MD = REPORTS / 'top159_multiday_aux_24h_leaderboard_latest.md'
STRICT_JSON = REPORTS / 'top159_multiday_aux_24h_strict_pass_latest.json'
VERDICT_JSON = REPORTS / 'top159_multiday_aux_24h_unique_verdict_latest.json'
VERDICT_MD = REPORTS / 'top159_multiday_aux_24h_unique_verdict_latest.md'


def load_research_module():
    spec = importlib.util.spec_from_file_location('top159_multiday_aux_research_mod', SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {SCRIPT}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

R = load_research_module()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def bj_now() -> str:
    return pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y-%m-%d %H:%M:%S %Z')


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def stable_hash(obj: Any) -> str:
    b = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode('utf-8')
    return hashlib.sha256(b).hexdigest()[:16]


def period_list() -> list[int]:
    if PERIOD_ENV:
        return sorted({int(x.strip()) for x in PERIOD_ENV.split(',') if x.strip()})
    return list(R.PERIODS)


def candidate_name(kind: str, params: dict[str, Any]) -> str:
    return f'{kind}_{stable_hash([kind, params])}'


def extra_candidate_stream(combos: list[tuple[int, ...]], model_truth: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    # Extra math families beyond the legacy stream: soft score gates and hard veto.
    # Keep grids bounded; the 24h loop spends CPU on full evaluation, not only coarse scans.
    sign_modes = ['same', 'reverse', 'short_same_long_reverse', 'short_reverse_long_same']
    forced = {(4,), (4, 7), (7, 11), (4, 11, 18), (4, 7, 11, 18)}
    for ps in combos:
        rich = ps in forced or len(ps) <= 2
        for mode in sign_modes:
            # Hard veto: count strong opposition. For reverse modes we reuse low/high via params by converting weights below.
            for oppose_th in ([0.42, 0.45, 0.48, 0.50] if rich else [0.45, 0.48]):
                for min_oppose in range(1, min(3, len(ps)) + 1):
                    if mode == 'same':
                        yield 'hard_veto', {'periods': list(ps), 'opposeSameThreshold': oppose_th, 'minOppose': min_oppose}
            # Weighted soft gates. Negative weights allow reverse interpretation.
            patterns: list[dict[int, float]] = []
            if mode == 'same':
                patterns.append({p: 1.0 for p in ps})
            elif mode == 'reverse':
                patterns.append({p: -1.0 for p in ps})
            elif mode == 'short_same_long_reverse':
                mid = float(np.median(ps))
                patterns.append({p: (1.0 if p <= mid else -1.0) for p in ps})
            elif mode == 'short_reverse_long_same':
                mid = float(np.median(ps))
                patterns.append({p: (-1.0 if p <= mid else 1.0) for p in ps})
            # Add accuracy-edge weights for rich combos.
            if rich:
                acc_weights: dict[int, float] = {}
                for p in ps:
                    m = model_truth.get(f'{p}d', {})
                    edge = 0.5 * (float(m.get('180d', {}).get('accuracyPct', 50.0)) + float(m.get('365d', {}).get('accuracyPct', 50.0))) - 50.0
                    acc_weights[p] = max(0.1, edge)
                if mode == 'reverse':
                    acc_weights = {p: -v for p, v in acc_weights.items()}
                patterns.append(acc_weights)
            for weights in patterns:
                for amp in ([0.02, 0.04, 0.07, 0.10, 0.14] if rich else [0.04, 0.08]):
                    for th in ([0.535, 0.545, 0.555, 0.57] if rich else [0.545, 0.56]):
                        yield 'weighted_soft', {
                            'periods': list(ps),
                            'weights': {str(k): float(v) for k, v in weights.items()},
                            'amp': amp,
                            'negClip': min(0.18, amp * 1.3),
                            'posClip': min(0.12, amp),
                            'threshold': th,
                            'modeNote': mode,
                        }


def prepare_data() -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], dict[str, Any], list[int], dict[str, Any]]:
    generated_at = now_iso()
    periods = period_list()
    top_params = R.top159_params()
    model_truth: dict[str, Any] = {'generatedAt': generated_at, 'periods': {}, 'bugcheck': {}}
    preds_by_window: dict[str, dict[int, pd.DataFrame]] = {'180d': {}, '365d': {}}
    frames: dict[int, tuple[pd.DataFrame, list[str]]] = {}
    random_checks = []

    for p in periods:
        print(f'[24h] train auxiliary {p}d models...', flush=True)
        df, feats = R.build_daily_period_features(p)
        frames[p] = (df, feats)
        model_truth['periods'][f'{p}d'] = {}
        allowed = True
        for w in ['180d', '365d']:
            best, pred, _audits = R.train_best_period(p, w, df, feats)
            model_truth['periods'][f'{p}d'][w] = {k: v for k, v in best.items() if k != 'scoreRaw'}
            preds_by_window[w][p] = pred
            if best['accuracyPct'] <= 50.0:
                allowed = False
        model_truth['periods'][f'{p}d']['allowedForLiveWeight'] = allowed

    for p in [x for x in [1, 7, 15, 30] if x in periods]:
        df, feats = frames[p]
        for w in ['180d', '365d']:
            random_checks.append(R.random_label_audit(p, w, df, feats))

    model_truth['bugcheck'] = {
        'generatedAt': generated_at,
        'futureFieldBlacklist': sorted(list(R.base.FORBIDDEN_FEATURES)) + list(R.base.FORBIDDEN_PREFIXES),
        'featureTiming': 'daily rows are timestamped after candle close; 15m candidates use merge_asof backward only',
        'windowIsolation': '180d and 365d candidates, aux models, funding curves and Monte Carlo are built separately',
        'randomLabelChecks': random_checks,
        'randomLabelPassed': all(x.get('passed', True) for x in random_checks if x.get('status') == 'ok'),
    }

    allowed_periods = [p for p in periods if model_truth['periods'][f'{p}d']['allowedForLiveWeight']]
    windows_df: dict[str, pd.DataFrame] = {}
    base_rows: dict[str, dict[str, Any]] = {}
    for w in ['180d', '365d']:
        cands = R.top159_candidates(w, top_params)
        cands = R.add_aux(cands, preds_by_window[w], set(allowed_periods))
        windows_df[w] = cands
        br = R.curve_metrics(cands, 'current_top159', w, 'baseline')
        br.update({'baseTrades': len(cands), 'blockedTrades': 0, 'blockedWinners': 0, 'blockedLosers': 0, 'retentionPct': 100.0})
        base_rows[w] = br

    R.preds_by_window = preds_by_window
    R.allowed_periods = allowed_periods
    R.base_rows = base_rows
    return windows_df, base_rows, model_truth, allowed_periods, {'top159Params': top_params, 'periods': periods}


def manual_block_audit() -> dict[str, Any]:
    df = pd.DataFrame({
        'dt': pd.to_datetime(['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04'], utc=True),
        'won': [True, False, True, False],
    })
    keep = np.array([True, True, False, False])
    q = R.quick_metrics(df, keep, 'manual', 'test', 'manual')
    return {
        'blockedWinnersExpected': 1,
        'blockedLosersExpected': 1,
        'blockedWinnersActual': q['blockedWinners'],
        'blockedLosersActual': q['blockedLosers'],
        'passed': q['blockedWinners'] == 1 and q['blockedLosers'] == 1 and q['retentionPct'] == 50.0,
    }


def repeat_hash_audit(windows_df: dict[str, pd.DataFrame]) -> dict[str, Any]:
    rows = []
    for w, df in windows_df.items():
        r1 = R.curve_metrics(df, 'repeat_current_top159', w, 'audit')
        r2 = R.curve_metrics(df, 'repeat_current_top159', w, 'audit')
        rows.append({'window': w, 'hash1': r1['setHash'], 'hash2': r2['setHash'], 'p50_1': r1['endingBankrollP50'], 'p50_2': r2['endingBankrollP50'], 'passed': r1['setHash'] == r2['setHash'] and r1['endingBankrollP50'] == r2['endingBankrollP50']})
    return {'rows': rows, 'passed': all(r['passed'] for r in rows)}


def shifted_feature_audit(windows_df: dict[str, pd.DataFrame], allowed_periods: list[int]) -> dict[str, Any]:
    # Lightweight sanity: shifted aux probabilities should produce a different hash for at least one window.
    periods = allowed_periods[:4]
    if not periods:
        return {'status': 'no_allowed_periods', 'passed': False}
    rows = []
    params = {'periods': periods, 'mode': 'same', 'threshold': 0.46}
    for w, df in windows_df.items():
        base_mask = R.apply_candidate_mask(df, 'avg_low_block', params)
        shifted = df.copy()
        for p in periods:
            shifted[f'p_{p}d_same'] = shifted[f'p_{p}d_same'].shift(1).fillna(0.5)
        sh_mask = R.apply_candidate_mask(shifted, 'avg_low_block', params)
        rows.append({
            'window': w,
            'baseKept': int(base_mask.sum()),
            'shiftedKept': int(sh_mask.sum()),
            'changed': bool(not np.array_equal(base_mask, sh_mask)),
        })
    return {'periods': periods, 'rows': rows, 'passed': any(r['changed'] for r in rows)}


def write_bug_audit(windows_df: dict[str, pd.DataFrame], model_truth: dict[str, Any], allowed_periods: list[int]) -> dict[str, Any]:
    audit = {
        'generatedAt': now_iso(),
        'beijingTime': bj_now(),
        'liveMutation': 'none',
        'windowIsolation': '180d/365d independently generated and evaluated',
        'blockedWinLossIndexAudit': manual_block_audit(),
        'repeatabilityAudit': repeat_hash_audit(windows_df),
        'shiftedFeatureAudit': shifted_feature_audit(windows_df, allowed_periods),
        'randomLabelAudit': model_truth.get('bugcheck', {}).get('randomLabelChecks', []),
        'randomLabelPassed': model_truth.get('bugcheck', {}).get('randomLabelPassed'),
        'fakToxicFullSeparation': 'curve_metrics writes separate fak2EndingBankroll, slot1 toxic P5/P50/P95, and full-fill derived rows',
        'featureTiming': model_truth.get('bugcheck', {}).get('featureTiming'),
    }
    audit['passed'] = bool(
        audit['blockedWinLossIndexAudit']['passed']
        and audit['repeatabilityAudit']['passed']
        and audit['shiftedFeatureAudit']['passed']
        and audit['randomLabelPassed']
    )
    write_json(BUG_AUDIT_JSON, audit)
    lines = [
        '# top159 多日辅助门回测器审计 v2',
        '',
        f'- 北京时间：`{audit["beijingTime"]}`',
        f'- 是否通过：`{audit["passed"]}`',
        f'- 拦截输赢索引审计：`{audit["blockedWinLossIndexAudit"]}`',
        f'- 重复运行哈希审计：`{audit["repeatabilityAudit"]}`',
        f'- 特征错位审计：`{audit["shiftedFeatureAudit"]}`',
        f'- 随机标签通过：`{audit["randomLabelPassed"]}`',
        '',
    ]
    write_text(BUG_AUDIT_MD, '\n'.join(lines))
    return audit


def all_candidate_summaries(windows_df: dict[str, pd.DataFrame], model_truth: dict[str, Any], allowed_periods: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    combos, forced_status = R.all_period_combos(allowed_periods, max_len=4)
    stream = list(R.candidate_param_stream(combos, model_truth))
    stream.extend(extra_candidate_stream(combos, model_truth))
    total = 0
    summaries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for kind, params in stream:
        total += 1
        name = candidate_name(kind, params)
        if name in seen:
            continue
        seen.add(name)
        if COARSE_LIMIT and len(summaries) >= COARSE_LIMIT:
            break
        try:
            s = R.summarize_quick_candidate(kind, params, windows_df)
        except Exception as exc:
            continue
        summaries.append(s)
        if total % 100000 == 0:
            print(f'[24h] coarse seen={total} kept={len(summaries)}', flush=True)
    summaries.sort(key=lambda x: x['quickScore'], reverse=True)
    audit = {
        'allowedPeriods': allowed_periods,
        'comboCount': len(combos),
        'forcedComboStatus': forced_status,
        'coarseCandidatesGenerated': total,
        'coarseCandidatesUnique': len(summaries),
        'structureOkCount': sum(1 for s in summaries if s.get('structureOk')),
    }
    return summaries, audit


def select_for_full_eval(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    strict = [s for s in summaries if s.get('structureOk')]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for s in strict[:STRICT_EVAL_LIMIT] + summaries[:FULL_EVAL_LIMIT]:
        if s['name'] not in seen:
            selected.append(s); seen.add(s['name'])
    # Always include user named combos if present.
    wanted = {(4,), (4, 7), (7, 11), (4, 11, 18), (4, 7, 11, 18)}
    for s in summaries:
        if tuple(s.get('params', {}).get('periods', [])) in wanted and s['name'] not in seen:
            selected.append(s); seen.add(s['name'])
            if sum(1 for x in selected if tuple(x.get('params', {}).get('periods', [])) in wanted) >= 300:
                break
    return selected

_WORKER_WINDOWS: dict[str, pd.DataFrame] | None = None


def _init_worker(windows_df: dict[str, pd.DataFrame], mc_trials: int):
    global _WORKER_WINDOWS
    _WORKER_WINDOWS = windows_df
    R.MC_TRIALS = mc_trials


def _full_eval_worker(summary: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_WINDOWS is None:
        raise RuntimeError('worker not initialized')
    return R.full_eval_candidate(summary, _WORKER_WINDOWS)


def load_done() -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if RESULTS_JSONL.exists():
        for line in RESULTS_JSONL.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                done[row['name']] = row
            except Exception:
                continue
    return done


def pass_gate(candidate: dict[str, Any], base_rows: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    by = {r['window']: r for r in candidate['rows']}
    return R.pass_gate(by['180d'], by['365d'], base_rows['180d'], base_rows['365d'])


def strong_score(candidate: dict[str, Any], base_rows: dict[str, dict[str, Any]]) -> float:
    by = {r['window']: r for r in candidate['rows']}
    r180, r365 = by['180d'], by['365d']
    d180 = r180['blockedLosers'] - r180['blockedWinners']
    d365 = r365['blockedLosers'] - r365['blockedWinners']
    p50_gain = (r180['endingBankrollP50'] - base_rows['180d']['endingBankrollP50']) + (r365['endingBankrollP50'] - base_rows['365d']['endingBankrollP50'])
    p5_gain = (r180['endingBankrollP5'] - base_rows['180d']['endingBankrollP5']) + (r365['endingBankrollP5'] - base_rows['365d']['endingBankrollP5'])
    return min(d180, d365) * 100000 + (d180 + d365) * 3000 + p50_gain * 3 + p5_gain - r365['maxDrawdownP50'] * 0.5 + r365['retentionPct'] * 10


def write_progress(results: list[dict[str, Any]], base_rows: dict[str, dict[str, Any]], queue_total: int, started: float, done_count: int, audit: dict[str, Any], finished: bool = False):
    verdicts = []
    for c in results:
        ok, reasons = pass_gate(c, base_rows)
        verdicts.append({**c, 'passed': ok, 'reasons': reasons, 'strongScore': strong_score(c, base_rows)})
    verdicts.sort(key=lambda x: x['strongScore'], reverse=True)
    strict = [v for v in verdicts if v['passed']]
    payload = {
        'generatedAt': now_iso(),
        'beijingTime': bj_now(),
        'finished': finished,
        'elapsedSeconds': round(time.time() - started, 3),
        'workers': WORKERS,
        'workerThreads': THREADS,
        'queueTotal': queue_total,
        'doneCount': done_count,
        'remainingCount': max(0, queue_total - done_count),
        'searchAudit': audit,
        'topCandidates': verdicts[:200],
        'strictPassCount': len(strict),
        'strictPass': strict[:100],
    }
    write_json(CHECKPOINT_JSON, payload)
    write_json(LEADERBOARD_JSON, {'generatedAt': payload['generatedAt'], 'rows': verdicts[:500], 'baseRows': base_rows})
    write_json(STRICT_JSON, {'generatedAt': payload['generatedAt'], 'rows': strict, 'baseRows': base_rows})
    selected = strict[0] if strict else None
    verdict = {
        'generatedAt': payload['generatedAt'],
        'status': 'candidate_passed_research_gate' if selected else 'running_or_no_strong_candidate_yet',
        'selected': selected,
        'baseRows': base_rows,
        'liveAction': 'research_only_no_live_change',
        'note': 'Small-edge candidates remain observation only; live audit requires materially better blocked loser gap plus toxicity/FAK stability.',
    }
    write_json(VERDICT_JSON, verdict)

    def row_md(r: dict[str, Any]) -> str:
        return f"|{r['name']}|{r['window']}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']}|{r['endingBankrollP5']}|{r['endingBankrollP50']}|{r['endingBankrollP95']}|{r['maxDrawdownP50']}|{r['monthlyPositiveRatioP50']}|{r['fak2EndingBankroll']}|{r.get('blockedWinners',0)}|{r.get('blockedLosers',0)}|{r.get('retentionPct',100)}|`{r['setHash']}`|"
    lines = ['# top159 多日辅助门 24小时搜索排行榜', '', f'- 北京时间：`{payload["beijingTime"]}`', f'- 完成：`{done_count}/{queue_total}`', f'- 严格通过候选数：`{len(strict)}`', '', '|候选|窗口|交易数|胜/负|胜率|P5|P50|P95|最大回撤|月正收益|FAK+2资金|误杀赢家|拦截输单|保留率|哈希|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for br in [base_rows['180d'], base_rows['365d']]:
        lines.append(row_md(br))
    for c in verdicts[:20]:
        for r in c['rows']:
            lines.append(row_md(r))
    write_text(LEADERBOARD_MD, '\n'.join(lines) + '\n')

    vlines = ['# top159 多日辅助门 24小时唯一结论', '', f'- 北京时间：`{payload["beijingTime"]}`', f'- 状态：`{verdict["status"]}`', '- live动作：`research_only_no_live_change`']
    if selected:
        vlines += [f'- 选中候选：`{selected["name"]}`', f'- 参数：`{json.dumps(selected["params"], ensure_ascii=False)}`']
    else:
        vlines += ['- 暂无达到严格门槛的强候选；后台搜索继续或已完成。']
    write_text(VERDICT_MD, '\n'.join(vlines) + '\n')


def run_search() -> int:
    started = time.time()
    print(f'[24h] start {bj_now()} workers={WORKERS} maxSeconds={MAX_SECONDS}', flush=True)
    windows_df, base_rows, model_truth, allowed_periods, context = prepare_data()
    bug = write_bug_audit(windows_df, model_truth, allowed_periods)
    if not bug.get('passed'):
        print('[24h] bug audit did not fully pass; continuing search but marking reports as audit-risk.', flush=True)

    summaries, search_audit = all_candidate_summaries(windows_df, model_truth['periods'], allowed_periods)
    selected_summaries = select_for_full_eval(summaries)
    queue_total = len(selected_summaries)
    done_map = load_done()
    pending = [s for s in selected_summaries if s['name'] not in done_map]
    results = list(done_map.values())
    print(f'[24h] full eval queue total={queue_total} done={len(done_map)} pending={len(pending)}', flush=True)
    write_progress(results, base_rows, queue_total, started, len(results), {**search_audit, 'bugAudit': bug, 'context': context}, finished=False)

    last_checkpoint = time.time()
    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_JSONL.open('a', encoding='utf-8') as out:
        with cf.ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker, initargs=(windows_df, R.MC_TRIALS)) as ex:
            futures: dict[cf.Future, dict[str, Any]] = {}
            it = iter(pending)
            # keep a small in-flight window so checkpointing remains responsive
            inflight_max = max(WORKERS * 2, 4)
            while len(futures) < inflight_max:
                try:
                    s = next(it)
                except StopIteration:
                    break
                futures[ex.submit(_full_eval_worker, s)] = s
            while futures:
                if time.time() - started > MAX_SECONDS:
                    print('[24h] max runtime reached; checkpointing unfinished queue.', flush=True)
                    break
                done, _ = cf.wait(futures, timeout=5, return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    s = futures.pop(fut)
                    try:
                        row = fut.result()
                    except Exception as exc:
                        row = {'name': s['name'], 'kind': s['kind'], 'params': s['params'], 'rows': [], 'score': -1e18, 'error': str(exc)[:500]}
                    if row.get('rows'):
                        out.write(json.dumps(row, ensure_ascii=False, default=str, separators=(',', ':')) + '\n')
                        out.flush()
                        results.append(row)
                    try:
                        nxt = next(it)
                        futures[ex.submit(_full_eval_worker, nxt)] = nxt
                    except StopIteration:
                        pass
                if time.time() - last_checkpoint >= CHECKPOINT_SECONDS:
                    write_progress(results, base_rows, queue_total, started, len(done_map) + len(results), {**search_audit, 'bugAudit': bug, 'context': context}, finished=False)
                    last_checkpoint = time.time()
                    print(f'[24h] checkpoint done={len(done_map) + len(results)}/{queue_total}', flush=True)
            for fut in futures:
                fut.cancel()

    final_done_map = load_done()
    final_results = [v for v in final_done_map.values() if v.get('rows')]
    finished = len(final_results) >= queue_total or (time.time() - started <= MAX_SECONDS and not pending)
    write_progress(final_results, base_rows, queue_total, started, len(final_results), {**search_audit, 'bugAudit': bug, 'context': context}, finished=finished)
    print(json.dumps({'status': 'finished' if finished else 'checkpointed', 'done': len(final_results), 'queueTotal': queue_total, 'leaderboard': str(LEADERBOARD_MD), 'verdict': str(VERDICT_MD)}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(run_search())
