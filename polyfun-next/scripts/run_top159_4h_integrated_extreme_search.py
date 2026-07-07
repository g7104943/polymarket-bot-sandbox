#!/usr/bin/env python3
from __future__ import annotations

# Research-only 4h integrated top159 extreme optimizer.
# Does not mutate live config and does not submit orders.

import concurrent.futures as cf
import hashlib
import importlib.util
import itertools
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault('TOP159_STACKED_WORKER_THREADS', os.environ.get('TOP159_4H_WORKER_THREADS', '2'))
for k in ['OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(k, os.environ.get('TOP159_4H_WORKER_THREADS', '2'))

import numpy as np
import pandas as pd

ROOT = Path('/Users/mac/polyfun')
REPORTS = ROOT / 'reports'
STACKED_SCRIPT = ROOT / 'polyfun-next' / 'scripts' / 'run_top159_stacked_aux_integrated_search.py'

WORKERS = int(os.environ.get('TOP159_4H_WORKERS', '4'))
MAX_SECONDS = int(os.environ.get('TOP159_4H_MAX_SECONDS', str(24 * 3600)))
MAX_MODEL_SPECS = int(os.environ.get('TOP159_4H_MAX_MODEL_SPECS', '0'))
QUICK_KEEP = int(os.environ.get('TOP159_4H_QUICK_KEEP', '1200'))
FULL_KEEP = int(os.environ.get('TOP159_4H_FULL_KEEP', '240'))
CHECKPOINT_SECONDS = int(os.environ.get('TOP159_4H_CHECKPOINT_SECONDS', '900'))
MC_TRIALS = int(os.environ.get('TOP159_4H_MC_TRIALS', '800'))
RNG_SEED = 9159

OUT_RESULTS = REPORTS / 'top159_4h_integrated_extreme_results_latest.jsonl'
OUT_CHECKPOINT = REPORTS / 'top159_4h_integrated_extreme_checkpoint_latest.json'
OUT_LEADERBOARD_MD = REPORTS / 'top159_4h_integrated_extreme_leaderboard_latest.md'
OUT_LEADERBOARD_JSON = REPORTS / 'top159_4h_integrated_extreme_leaderboard_latest.json'
OUT_VERDICT_MD = REPORTS / 'top159_4h_integrated_extreme_unique_verdict_latest.md'
OUT_VERDICT_JSON = REPORTS / 'top159_4h_integrated_extreme_unique_verdict_latest.json'
OUT_AUDIT = REPORTS / 'top159_4h_integrated_extreme_bug_audit_latest.json'


def clear_latest_outputs_if_requested() -> None:
    """Avoid mixing smoke-test rows with the formal latest run."""
    if os.environ.get('TOP159_4H_RESUME', '0') == '1':
        return
    for path in (
        OUT_RESULTS,
        OUT_CHECKPOINT,
        OUT_LEADERBOARD_JSON,
        OUT_LEADERBOARD_MD,
        OUT_VERDICT_JSON,
        OUT_VERDICT_MD,
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

S = load_module('top159_stacked_base_for_4h_extreme', STACKED_SCRIPT)
S.base.TRIALS = MC_TRIALS
S.aux.MC_TRIALS = MC_TRIALS


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def bj_now() -> str:
    return datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + '\n')


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode('utf-8')).hexdigest()[:16]


def split_train_val(df: pd.DataFrame, window: str, train_window: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = df['dt'].max()
    days = 180 if window == '180d' else 365
    start = end - pd.Timedelta(days=days)
    val = df[df['dt'] >= start].copy()
    if train_window == 'full':
        train = df[df['dt'] < start].copy()
    else:
        train_days = {'1y': 365, '2y': 730, '3y': 1095, '4y': 1460, '5y': 1825}.get(train_window)
        if train_days is None:
            raise ValueError(f'unsupported train_window {train_window}')
        train = df[(df['dt'] < start) & (df['dt'] >= start - pd.Timedelta(days=train_days))].copy()
    max_rows = int(os.environ.get('TOP159_STACKED_MAX_TRAIN_ROWS', '0') or 0)
    if max_rows > 0 and len(train) > max_rows:
        train = train.sort_values('dt').iloc[-max_rows:].copy()
    return train, val


def feature_subset(all_features: list[str], base_mode: str) -> list[str]:
    combo = ('4h',)
    mode = 'same'
    if base_mode == 'prob_only':
        base_feats: list[str] = []
    elif base_mode == 'base15_core':
        wanted = {
            'ret_1','ret_2','ret_4','ret_8','ret_16','ret_32','vol_8','vol_16','vol_32','range_8','range_16',
            'ema_8_32','ema_16_64','ema_dist_8','ema_dist_16','ema_dist_32','rsi_14','bb_pos','hour_sin','hour_cos','dow_sin','dow_cos'
        }
        base_feats = [c for c in all_features if c in wanted]
    else:
        base_feats = [c for c in all_features if not c.startswith(('p_', 'top159_', 'combo_', '1h_', '4h_', '4d_', '7d_', '18d_'))]
    feats = base_feats + S.combo_features(combo, mode)
    out = []
    for c in feats:
        if c in all_features and c not in out:
            out.append(c)
    return out[:120]


def model_specs() -> list[dict[str, Any]]:
    train_windows = ['1y', '2y', '3y', '4y', '5y', 'full']
    base_modes = ['prob_only', 'base15_core', 'base15_all']
    specs: list[dict[str, Any]] = []
    for tw, bm in itertools.product(train_windows, base_modes):
        for C in [0.15, 0.25, 0.4, 0.65, 1.0, 1.6, 2.5, 4.0, 6.5, 10.0]:
            specs.append({'engine': 'logistic', 'train_window': tw, 'base_mode': bm, 'C': C, 'n_estimators': 1, 'learning_rate': 0.04, 'num_leaves': 16, 'min_child_samples': 80, 'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_lambda': 1.0, 'depth': 3})
        tree_grid = [
            (80, 0.020, 16, 60, 0.4, 3), (120, 0.025, 24, 80, 0.8, 4),
            (160, 0.030, 32, 100, 1.2, 4), (220, 0.035, 36, 80, 1.6, 4),
            (260, 0.020, 48, 120, 2.5, 5), (180, 0.045, 24, 60, 0.8, 3),
            (120, 0.055, 16, 120, 3.5, 3), (260, 0.030, 64, 160, 4.5, 5),
        ]
        for engine in ['lightgbm', 'xgboost', 'catboost']:
            for ne, lr, leaves, mcs, reg, depth in tree_grid:
                specs.append({'engine': engine, 'train_window': tw, 'base_mode': bm, 'C': 1.0, 'n_estimators': ne, 'learning_rate': lr, 'num_leaves': leaves, 'min_child_samples': mcs, 'subsample': 0.88, 'colsample_bytree': 0.88, 'reg_lambda': reg, 'depth': depth})
    # Prioritize known neighborhood, but do not exclude the rest.
    def rank(s: dict[str, Any]) -> tuple[int, str]:
        pri = 0
        if s['engine'] == 'logistic': pri -= 20
        if s['train_window'] == '3y': pri -= 10
        if s['base_mode'] == 'base15_core': pri -= 5
        if abs(float(s.get('C', 1.0)) - 1.0) < 1e-9: pri -= 3
        return pri, json.dumps(s, sort_keys=True)
    specs.sort(key=rank)
    if MAX_MODEL_SPECS > 0:
        specs = specs[:MAX_MODEL_SPECS]
    return specs


EDGES = [round(x, 3) for x in np.arange(0.020, 0.0901, 0.005)]
MIN_TOP159 = [0.535, 0.545, 0.555, 0.565]
COMBO_MIN_SAME = [0.0, 0.48, 0.50, 0.52, 0.55]


def threshold_params_for_spec(spec: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for edge, mts, cms in itertools.product(EDGES, MIN_TOP159, COMBO_MIN_SAME):
        p = dict(spec)
        p.update({'combo': ('4h',), 'combo_mode': 'same', 'edge': edge, 'min_top159_score': mts, 'combo_min_same': cms})
        out.append(p)
    return out


def quick_metrics(selected: pd.DataFrame, name: str, window: str, base_trades: int) -> dict[str, Any]:
    trades = int(len(selected))
    wins = int(selected['won'].sum()) if trades else 0
    losses = trades - wins
    wr = 100.0 * wins / trades if trades else 0.0
    ret = 100.0 * trades / base_trades if base_trades else 0.0
    # Simple conservative Kelly-like proxy: 1% stake, win +1, loss -1.
    bank = 850.0
    peak = bank
    dd = 0.0
    for won in selected['won'].astype(bool).to_numpy():
        stake = bank * 0.01
        bank += stake if won else -stake
        peak = max(peak, bank)
        dd = max(dd, peak - bank)
    return {'name': name, 'window': window, 'trades': trades, 'wins': wins, 'losses': losses, 'winRatePct': round(wr, 6), 'quickEndingBankroll': round(bank, 6), 'quickDrawdown': round(dd, 6), 'retentionPct': round(ret, 6)}


def selected_rows(val: pd.DataFrame, prob: np.ndarray, params: dict[str, Any]) -> pd.DataFrame:
    pred_up = prob >= 0.5
    score = np.maximum(prob, 1.0 - prob)
    mask = score >= 0.5 + float(params['edge'])
    if params.get('min_top159_score', 0.0) > 0:
        mask &= pd.to_numeric(val['top159_oof_score'], errors='coerce').fillna(0.5).to_numpy() >= float(params['min_top159_score'])
    if params.get('combo_min_same', 0.0) > 0:
        mask &= pd.to_numeric(val['combo_same_mean_same'], errors='coerce').fillna(0.5).to_numpy() >= float(params['combo_min_same'])
    out = val.loc[mask, ['dt', 'label_up']].copy().reset_index(drop=True)
    if len(out) == 0:
        return out.assign(pred_up15=[], score15=[], won=[])
    out['pred_up15'] = pred_up[mask].astype(bool)
    out['score15'] = score[mask]
    out['won'] = out['pred_up15'].to_numpy() == out['label_up'].astype(bool).to_numpy()
    return out


_DF: pd.DataFrame | None = None
_FEATURES: list[str] | None = None
_BASE: dict[str, dict[str, Any]] | None = None


def init_worker(df: pd.DataFrame, features: list[str], base_rows: dict[str, dict[str, Any]]) -> None:
    global _DF, _FEATURES, _BASE
    _DF = df
    _FEATURES = features
    _BASE = base_rows
    S.base.TRIALS = MC_TRIALS
    S.aux.MC_TRIALS = MC_TRIALS


def eval_spec_worker(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    if _DF is None or _FEATURES is None or _BASE is None:
        raise RuntimeError('worker not initialized')
    idx, spec = item
    feats = feature_subset(_FEATURES, spec['base_mode'])
    if len(feats) < 8:
        return {'idx': idx, 'spec': spec, 'error': 'too_few_features'}
    probs: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    fit_errors = []
    for window in ['180d', '365d']:
        train, val = split_train_val(_DF, window, spec['train_window'])
        try:
            model = S.fit_main(spec['engine'], train, feats, {**spec, 'combo': ('4h',), 'combo_mode': 'same', 'edge': 0.06, 'min_top159_score': 0.545, 'combo_min_same': 0.0})
            if model is None:
                return {'idx': idx, 'spec': spec, 'error': f'fit_failed_{window}'}
            prob = S.predict_prob(model, val, feats)
            probs[window] = (val, prob)
        except Exception as exc:
            fit_errors.append(f'{window}:{str(exc)[:180]}')
            return {'idx': idx, 'spec': spec, 'error': ';'.join(fit_errors)}
    quick = []
    for p in threshold_params_for_spec(spec):
        qrows = []
        ok = True
        for window in ['180d', '365d']:
            val, prob = probs[window]
            sel = selected_rows(val, prob, p)
            qm = quick_metrics(sel, 'quick', window, int(_BASE[window]['trades']))
            qrows.append(qm)
            if qm['trades'] < max(100 if window == '180d' else 200, int(_BASE[window]['trades'] * 0.35)):
                ok = False
        if not ok:
            continue
        by = {r['window']: r for r in qrows}
        # Risk-first quick score. This is only for screening.
        score = (
            min(by['180d']['quickEndingBankroll'] - 850.0, by['365d']['quickEndingBankroll'] - 850.0) * 3.0
            + (by['180d']['winRatePct'] + by['365d']['winRatePct']) * 8.0
            + min(by['180d']['retentionPct'], by['365d']['retentionPct']) * 3.0
            - by['365d']['quickDrawdown'] * 0.8
        )
        quick.append({'params': p, 'quickRows': qrows, 'quickScore': score})
    quick.sort(key=lambda x: x['quickScore'], reverse=True)
    return {'idx': idx, 'spec': spec, 'featureCount': len(feats), 'quickTop': quick[:10]}


def full_eval_worker(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    if _DF is None or _FEATURES is None or _BASE is None:
        raise RuntimeError('worker not initialized')
    idx, params = item
    try:
        out = full_eval(_DF, _FEATURES, params, _BASE)
        return {'idx': idx, 'params': params, 'result': out}
    except Exception as exc:
        return {'idx': idx, 'params': params, 'error': str(exc)[:500]}


def full_eval(df: pd.DataFrame, features: list[str], params: dict[str, Any], base_rows: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    # Reuse the production evaluator but through the local split monkeypatch to allow 4y.
    old_split = S.split_train_val
    S.split_train_val = split_train_val
    try:
        name = f"4h_extreme_{params['engine']}_{params['train_window']}_{params['base_mode']}_edge{params['edge']}_mts{params['min_top159_score']}_cms{params['combo_min_same']}_{stable_hash(params)}"
        out = S.evaluate_params(df, features, params, name)
        if out is None:
            return None
        ok, reasons = S.pass_gate(out, base_rows)
        return {**out, 'passed': ok, 'reasons': reasons, 'score': S.score_candidate(out, base_rows)}
    finally:
        S.split_train_val = old_split


def write_reports(payload: dict[str, Any], base_rows: dict[str, dict[str, Any]]) -> None:
    candidates = payload.get('fullCandidates', [])
    candidates = sorted(candidates, key=lambda x: x.get('score', -1e18), reverse=True)
    strict = [c for c in candidates if c.get('passed')]
    selected = strict[0] if strict else (candidates[0] if candidates else None)
    report = {**payload, 'fullCandidates': candidates[:200], 'strictPassCount': len(strict), 'selected': selected, 'baseRows': base_rows}
    write_json(OUT_LEADERBOARD_JSON, report)
    verdict = {'generatedAt': payload['generatedAt'], 'beijingTime': bj_now(), 'status': 'candidate_passed_research_gate' if (selected and selected.get('passed')) else 'no_strict_candidate_yet', 'selected': selected, 'baseRows': base_rows, 'liveAction': 'research_only_no_live_change'}
    write_json(OUT_VERDICT_JSON, verdict)
    lines = ['# top159 4小时一体化极限优化', '', f"- 北京时间：`{bj_now()}`", f"- 完整复核候选：`{len(candidates)}`", f"- 严格通过：`{len(strict)}`", '- live动作：`research_only_no_live_change`', '', '|候选|窗口|模型|训练窗|特征组|edge|top159最低|4h支持|交易数|胜/负|胜率|P5|P50|P95|最大回撤|月正收益|FAK+2资金|哈希|', '|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for b in base_rows.values():
        lines.append(f"|current_top159|{b['window']}|baseline|-|-|-|-|-|{b['trades']}|{b['wins']}/{b['losses']}|{b['winRatePct']}|{b['endingBankrollP5']}|{b['endingBankrollP50']}|{b['endingBankrollP95']}|{b['maxDrawdownP50']}|{b['monthlyPositiveRatioP50']}|{b['fak2EndingBankroll']}|`{b['setHash']}`|")
    for c in candidates[:60]:
        p = c['params']
        for r in c['rows']:
            lines.append(f"|{c['name']}|{r['window']}|{p['engine']}|{p['train_window']}|{p['base_mode']}|{p['edge']}|{p['min_top159_score']}|{p['combo_min_same']}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']}|{r['endingBankrollP5']}|{r['endingBankrollP50']}|{r['endingBankrollP95']}|{r['maxDrawdownP50']}|{r['monthlyPositiveRatioP50']}|{r['fak2EndingBankroll']}|`{r['setHash']}`|")
    OUT_LEADERBOARD_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    vlines = ['# top159 4小时一体化极限优化唯一结论', '', f"- 北京时间：`{bj_now()}`", f"- 状态：`{verdict['status']}`", '- live动作：`research_only_no_live_change`']
    if selected:
        vlines += [f"- 选中候选：`{selected['name']}`", f"- 参数：`{json.dumps(selected['params'], ensure_ascii=False)}`", '', '|窗口|交易数|胜/负|胜率|P5|P50|P95|最大回撤|月正收益|FAK+2资金|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in selected['rows']:
            vlines.append(f"|{r['window']}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']}|{r['endingBankrollP5']}|{r['endingBankrollP50']}|{r['endingBankrollP95']}|{r['maxDrawdownP50']}|{r['monthlyPositiveRatioP50']}|{r['fak2EndingBankroll']}|")
    OUT_VERDICT_MD.write_text('\n'.join(vlines) + '\n', encoding='utf-8')


def main() -> int:
    start = time.time()
    clear_latest_outputs_if_requested()
    print(f'[4h-extreme] start {bj_now()} workers={WORKERS} maxSeconds={MAX_SECONDS}', flush=True)
    df, features, data_truth = S.build_stacked_frame()
    df = S.add_combo_columns(df, ('4h',), 'same')
    base_rows = S.base_rows()
    audit = S.bug_audit(df, features, base_rows, data_truth)
    audit['script'] = str(Path(__file__))
    audit['searchPolicy'] = '4h only; quick screen all thresholds, full MC for top candidates; risk-first scoring.'
    write_json(OUT_AUDIT, audit)
    if not audit.get('passed'):
        print('[4h-extreme] audit failed; aborting', flush=True)
        return 2

    specs = model_specs()
    total_specs = len(specs)
    print(f'[4h-extreme] rows={len(df)} features={len(features)} specs={total_specs}', flush=True)
    quick_pool: list[dict[str, Any]] = []
    done_specs = 0
    last_ck = time.time()
    with cf.ProcessPoolExecutor(max_workers=WORKERS, initializer=init_worker, initargs=(df, features, base_rows)) as ex:
        futures = {ex.submit(eval_spec_worker, (i, spec)): (i, spec) for i, spec in enumerate(specs)}
        for fut in cf.as_completed(futures):
            done_specs += 1
            try:
                res = fut.result()
            except Exception as exc:
                res = {'idx': futures[fut][0], 'spec': futures[fut][1], 'error': str(exc)[:400]}
            for q in res.get('quickTop', []) or []:
                quick_pool.append(q)
            if len(quick_pool) > QUICK_KEEP * 4:
                quick_pool.sort(key=lambda x: x['quickScore'], reverse=True)
                quick_pool = quick_pool[:QUICK_KEEP * 2]
            if time.time() - last_ck >= CHECKPOINT_SECONDS:
                quick_pool.sort(key=lambda x: x['quickScore'], reverse=True)
                write_json(OUT_CHECKPOINT, {'generatedAt': now_iso(), 'beijingTime': bj_now(), 'phase': 'quick_screen', 'doneSpecs': done_specs, 'totalSpecs': total_specs, 'quickPool': quick_pool[:QUICK_KEEP], 'baseRows': base_rows})
                print(f'[4h-extreme] quick checkpoint {done_specs}/{total_specs} pool={len(quick_pool)}', flush=True)
                last_ck = time.time()
            if time.time() - start > MAX_SECONDS * 0.55:
                print('[4h-extreme] time budget reached during quick stage; moving to full stage', flush=True)
                break
        for fut in futures:
            fut.cancel()

    quick_pool.sort(key=lambda x: x['quickScore'], reverse=True)
    selected_quick = quick_pool[:FULL_KEEP]
    print(f'[4h-extreme] full candidates={len(selected_quick)}', flush=True)
    full: list[dict[str, Any]] = []
    seen = set()
    full_inputs: list[dict[str, Any]] = []
    for q in selected_quick:
        p = q['params']
        key = stable_hash(p)
        if key in seen:
            continue
        seen.add(key)
        full_inputs.append(p)
    done_full = 0
    with cf.ProcessPoolExecutor(max_workers=WORKERS, initializer=init_worker, initargs=(df, features, base_rows)) as ex:
        futures = {ex.submit(full_eval_worker, (i, p)): p for i, p in enumerate(full_inputs)}
        for fut in cf.as_completed(futures):
            done_full += 1
            try:
                res = fut.result()
            except Exception as exc:
                res = {'error': str(exc)[:500], 'params': futures[fut]}
            out = res.get('result') if isinstance(res, dict) else None
            if out:
                full.append(out)
                append_jsonl(OUT_RESULTS, out)
            elif isinstance(res, dict) and res.get('error'):
                append_jsonl(OUT_RESULTS, {'error': res.get('error'), 'params': res.get('params')})
            if done_full % 20 == 0 or time.time() - last_ck >= CHECKPOINT_SECONDS:
                payload = {'generatedAt': now_iso(), 'beijingTime': bj_now(), 'phase': 'full_eval', 'doneFull': done_full, 'totalFull': len(full_inputs), 'quickDoneSpecs': done_specs, 'totalSpecs': total_specs, 'fullCandidates': full, 'baseRows': base_rows}
                write_json(OUT_CHECKPOINT, payload)
                write_reports(payload, base_rows)
                print(f'[4h-extreme] full checkpoint {done_full}/{len(full_inputs)} full={len(full)}', flush=True)
                last_ck = time.time()
            if time.time() - start > MAX_SECONDS:
                print('[4h-extreme] max seconds reached', flush=True)
                for pending in futures:
                    pending.cancel()
                break

    payload = {'generatedAt': now_iso(), 'beijingTime': bj_now(), 'phase': 'finished', 'doneSpecs': done_specs, 'totalSpecs': total_specs, 'fullCandidates': full, 'baseRows': base_rows, 'quickTopCount': len(full_inputs)}
    write_json(OUT_CHECKPOINT, payload)
    write_reports(payload, base_rows)
    print(json.dumps({'status': 'finished', 'doneSpecs': done_specs, 'totalSpecs': total_specs, 'fullCandidates': len(full), 'leaderboard': str(OUT_LEADERBOARD_MD), 'verdict': str(OUT_VERDICT_MD)}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
