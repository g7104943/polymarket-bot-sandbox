#!/usr/bin/env python3
from __future__ import annotations

# Fast two-stage search for top159 stacked auxiliary integration.
# Research-only. Reuses cached OOF features from the full stacked script.

import concurrent.futures as cf
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

THREADS = os.environ.get('TOP159_STACKED_FAST_WORKER_THREADS', '2')
for k in ['OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(k, THREADS)

import numpy as np
import pandas as pd

ROOT = Path('/Users/mac/polyfun')
REPORTS = ROOT / 'reports'
STACKED_SCRIPT = ROOT / 'polyfun-next' / 'scripts' / 'run_top159_stacked_aux_integrated_search.py'

WORKERS = int(os.environ.get('TOP159_STACKED_FAST_WORKERS', '4'))
MAX_SECONDS = int(os.environ.get('TOP159_STACKED_FAST_MAX_SECONDS', str(6 * 3600)))
FAST_FULL_EVAL_LIMIT = int(os.environ.get('TOP159_STACKED_FAST_FULL_EVAL_LIMIT', '80'))
FAST_TOP_COMBOS = int(os.environ.get('TOP159_STACKED_FAST_TOP_COMBOS', '8'))
FAST_MODEL_LIMIT = int(os.environ.get('TOP159_STACKED_FAST_MODEL_LIMIT', '1600'))
CHECKPOINT_SECONDS = int(os.environ.get('TOP159_STACKED_FAST_CHECKPOINT_SECONDS', '600'))

OUT_STAGE1_JSON = REPORTS / 'top159_stacked_aux_integrated_fast_stage1_latest.json'
OUT_RESULTS = REPORTS / 'top159_stacked_aux_integrated_fast_results_latest.jsonl'
OUT_CHECKPOINT = REPORTS / 'top159_stacked_aux_integrated_fast_checkpoint_latest.json'
OUT_LEADERBOARD_JSON = REPORTS / 'top159_stacked_aux_integrated_fast_leaderboard_latest.json'
OUT_LEADERBOARD_MD = REPORTS / 'top159_stacked_aux_integrated_fast_leaderboard_latest.md'
OUT_VERDICT_JSON = REPORTS / 'top159_stacked_aux_integrated_fast_unique_verdict_latest.json'
OUT_VERDICT_MD = REPORTS / 'top159_stacked_aux_integrated_fast_unique_verdict_latest.md'
OUT_AUDIT_JSON = REPORTS / 'top159_stacked_aux_integrated_fast_bug_audit_latest.json'


def load_stacked():
    spec = importlib.util.spec_from_file_location('stacked_full_for_fast', STACKED_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {STACKED_SCRIPT}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

S = load_stacked()
S.WORKERS = WORKERS
S.THREADS = THREADS
S.MC_TRIALS = int(os.environ.get('TOP159_STACKED_FAST_MC_TRIALS', '500'))
S.base.TRIALS = S.MC_TRIALS
S.aux.MC_TRIALS = S.MC_TRIALS


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


def val_window(df: pd.DataFrame, window: str) -> pd.DataFrame:
    end = df['dt'].max()
    days = 180 if window == '180d' else 365
    return df[df['dt'] >= end - pd.Timedelta(days=days)].copy().reset_index(drop=True)


def combo_score_array(df: pd.DataFrame, combo: tuple[str, ...], mode: str) -> np.ndarray:
    vals = []
    rank = {'1h': 1, '4h': 2, '4d': 3, '7d': 4, '18d': 5}
    for key in combo:
        same = pd.to_numeric(df[f'p_{key}_same_top159'], errors='coerce').fillna(0.5).to_numpy(float)
        if mode == 'reverse':
            same = 1.0 - same
        elif mode == 'short_same_long_reverse' and rank[key] >= 4:
            same = 1.0 - same
        elif mode == 'short_reverse_long_same' and rank[key] <= 3:
            same = 1.0 - same
        vals.append(same)
    return np.vstack(vals).mean(axis=0) if vals else np.full(len(df), 0.5)


def stage1_mask(df: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
    score = combo_score_array(df, tuple(params['combo']), params['mode'])
    support = 2.0 * score - 1.0
    final_score = pd.to_numeric(df['top159_oof_score'], errors='coerce').fillna(0.5).to_numpy(float) + np.clip(params['amp'] * support, -params['amp'] * 1.3, params['amp'])
    mask = final_score >= params['threshold']
    if params['min_top159_score'] > 0:
        mask &= pd.to_numeric(df['top159_oof_score'], errors='coerce').fillna(0.5).to_numpy(float) >= params['min_top159_score']
    if params['combo_min_same'] > 0:
        mask &= score >= params['combo_min_same']
    return mask


def quick_row(base_df: pd.DataFrame, mask: np.ndarray, name: str, window: str, params: dict[str, Any]) -> dict[str, Any]:
    pred = base_df['top159_oof_pred_up'].astype(bool).to_numpy()
    label = base_df['label_up'].astype(bool).to_numpy()
    won = pred == label
    kept = won[mask]
    blocked = won[~mask]
    return {
        'name': name, 'window': window, 'trades': int(mask.sum()), 'wins': int(kept.sum()), 'losses': int((~kept).sum()),
        'winRatePct': round(float(kept.mean()) * 100, 6) if len(kept) else 0.0,
        'blockedTrades': int((~mask).sum()), 'blockedWinners': int(blocked.sum()) if len(blocked) else 0,
        'blockedLosers': int((~blocked).sum()) if len(blocked) else 0,
        'retentionPct': round(float(mask.mean()) * 100, 6) if len(mask) else 0.0,
        'params': params,
    }


def stage1_candidates(df: pd.DataFrame, base_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    windows = {'180d': val_window(df, '180d'), '365d': val_window(df, '365d')}
    modes = ['same', 'reverse', 'short_same_long_reverse', 'short_reverse_long_same']
    rows = []
    for combo in S.all_combos():
        for mode in modes:
            for amp in [0.02, 0.04, 0.07, 0.10]:
                for threshold in [0.535, 0.545, 0.555, 0.57]:
                    for min_top in [0.0, 0.545]:
                        for combo_min in [0.0, 0.43, 0.47]:
                            params = {'combo': combo, 'mode': mode, 'amp': amp, 'threshold': threshold, 'min_top159_score': min_top, 'combo_min_same': combo_min}
                            name = 'fast_formula_' + S.stable_hash(params)
                            qrows = []
                            for w, wdf in windows.items():
                                mask = stage1_mask(wdf, params)
                                qrows.append(quick_row(wdf, mask, name, w, params))
                            by = {r['window']: r for r in qrows}
                            # Structure score first; full metrics are reserved for top candidates.
                            d180 = by['180d']['blockedLosers'] - by['180d']['blockedWinners']
                            d365 = by['365d']['blockedLosers'] - by['365d']['blockedWinners']
                            min_trades_penalty = max(0, 100 - by['180d']['trades']) * 20 + max(0, 200 - by['365d']['trades']) * 15
                            retention_penalty = max(0.0, 45.0 - by['365d']['retentionPct']) * 40
                            score = min(d180, d365) * 10000 + (d180 + d365) * 200 + by['180d']['winRatePct'] + by['365d']['winRatePct'] - min_trades_penalty - retention_penalty
                            rows.append({'name': name, 'kind': 'fast_formula', 'params': params, 'quickRows': qrows, 'quickScore': score, 'structureOk': d180 > 0 and d365 > 0 and by['180d']['trades'] >= 100 and by['365d']['trades'] >= 200 and by['365d']['retentionPct'] >= 45})
    rows.sort(key=lambda x: x['quickScore'], reverse=True)
    return rows


def full_eval_formula(df: pd.DataFrame, summary: dict[str, Any]) -> dict[str, Any]:
    rows = []
    params = summary['params']
    for w in ['180d', '365d']:
        wdf = val_window(df, w)
        mask = stage1_mask(wdf, params)
        out = pd.DataFrame({
            'dt': wdf.loc[mask, 'dt'].reset_index(drop=True),
            'label_up': wdf.loc[mask, 'label_up'].astype(bool).to_numpy(),
            'pred_up15': wdf.loc[mask, 'top159_oof_pred_up'].astype(bool).to_numpy(),
        })
        out['won'] = out['pred_up15'].to_numpy() == out['label_up'].to_numpy()
        row = S.aux.curve_metrics(out, summary['name'], w, 'fast_formula_full_eval')
        row.update({'combo': '+'.join(params['combo']), 'engine': 'formula', 'trainWindow': '-', 'edge': params['threshold'], 'featureCount': len(params['combo']), 'baseMode': 'formula', 'retentionPct': summary['quickRows'][0 if w == '180d' else 1]['retentionPct'], 'blockedWinners': summary['quickRows'][0 if w == '180d' else 1]['blockedWinners'], 'blockedLosers': summary['quickRows'][0 if w == '180d' else 1]['blockedLosers']})
        rows.append(row)
    return {'name': summary['name'], 'params': params, 'rows': rows, 'score': S.score_candidate({'rows': rows}, BASE_ROWS)}


def derive_model_params(stage1: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combo_modes = []
    seen = set()
    for row in stage1:
        params = row['params']
        key = (tuple(params['combo']), params['mode'])
        if key not in seen:
            combo_modes.append(key)
            seen.add(key)
        if len(combo_modes) >= FAST_TOP_COMBOS:
            break
    # Always keep a couple of explicitly important combos if present.
    for key in [(('4d',), 'same'), (('1h','4d'), 'same'), (('1h','4h','4d','7d','18d'), 'same')]:
        if key not in seen:
            combo_modes.append(key); seen.add(key)
    params_list = []
    engines = [e for e in ['lightgbm', 'xgboost', 'logistic'] if e in S.ENGINES or True]
    for combo, mode in combo_modes:
        for engine in engines:
            for train_window in ['1y', '3y', '5y']:
                for base_mode in ['prob_only', 'base15_core']:
                    for edge in [0.03, 0.045, 0.06, 0.075]:
                        for mts in [0.0, 0.545]:
                            for cms in [0.0, 0.47]:
                                if engine == 'logistic':
                                    params_list.append({'engine': engine, 'train_window': train_window, 'combo': combo, 'combo_mode': mode, 'base_mode': base_mode, 'edge': edge, 'min_top159_score': mts, 'combo_min_same': cms, 'n_estimators': 1, 'learning_rate': 0.04, 'num_leaves': 16, 'min_child_samples': 80, 'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_lambda': 1.0, 'depth': 3, 'C': 1.0})
                                else:
                                    params_list.append({'engine': engine, 'train_window': train_window, 'combo': combo, 'combo_mode': mode, 'base_mode': base_mode, 'edge': edge, 'min_top159_score': mts, 'combo_min_same': cms, 'n_estimators': 120, 'learning_rate': 0.035, 'num_leaves': 24, 'min_child_samples': 80, 'subsample': 0.88, 'colsample_bytree': 0.88, 'reg_lambda': 0.8, 'depth': 4})
    return params_list[:FAST_MODEL_LIMIT]

BASE_ROWS: dict[str, dict[str, Any]] = {}
_DF: pd.DataFrame | None = None
_FEATURES: list[str] | None = None


def init_worker(df: pd.DataFrame, features: list[str], base_rows: dict[str, dict[str, Any]]):
    global _DF, _FEATURES, BASE_ROWS
    _DF = df
    _FEATURES = features
    BASE_ROWS = base_rows
    S.base.TRIALS = S.MC_TRIALS
    S.aux.MC_TRIALS = S.MC_TRIALS


def eval_model_worker(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    if _DF is None or _FEATURES is None:
        raise RuntimeError('worker not initialized')
    i, params = item
    name = f"fast_model_{params['engine']}_{params['train_window']}_{'+'.join(params['combo'])}_{params['combo_mode']}_{params['base_mode']}_edge{params['edge']}_{S.stable_hash(params)}"
    try:
        out = S.evaluate_params(_DF, _FEATURES, params, name)
        if out is None:
            return {'name': name, 'params': params, 'error': 'empty_or_fit_failed'}
        return out
    except Exception as exc:
        return {'name': name, 'params': params, 'error': str(exc)[:500]}


def pass_gate(candidate: dict[str, Any], base_rows: dict[str, dict[str, Any]]):
    return S.pass_gate(candidate, base_rows)


def score_candidate(candidate: dict[str, Any], base_rows: dict[str, dict[str, Any]]) -> float:
    return S.score_candidate(candidate, base_rows)


def write_outputs(results: list[dict[str, Any]], base_rows: dict[str, dict[str, Any]], total: int, started: float, data_truth: dict[str, Any], finished: bool):
    valid = [r for r in results if r.get('rows')]
    verdicts = []
    for c in valid:
        ok, reasons = pass_gate(c, base_rows)
        verdicts.append({**c, 'passed': ok, 'reasons': reasons, 'score': score_candidate(c, base_rows)})
    verdicts.sort(key=lambda x: x['score'], reverse=True)
    strict = [v for v in verdicts if v['passed']]
    payload = {'generatedAt': now_iso(), 'beijingTime': bj_now(), 'finished': finished, 'elapsedSeconds': round(time.time()-started,3), 'totalCandidates': total, 'doneCount': len(results), 'validCount': len(valid), 'strictPassCount': len(strict), 'workers': WORKERS, 'workerThreads': THREADS, 'baseRows': base_rows, 'dataTruth': data_truth, 'topCandidates': verdicts[:300], 'strictPass': strict[:100]}
    write_json(OUT_CHECKPOINT, payload)
    write_json(OUT_LEADERBOARD_JSON, {'generatedAt': payload['generatedAt'], 'baseRows': base_rows, 'rows': verdicts[:500]})
    selected = strict[0] if strict else None
    verdict = {'generatedAt': payload['generatedAt'], 'status': 'candidate_passed_research_gate' if selected else 'running_or_no_fast_candidate_yet', 'selected': selected, 'baseRows': base_rows, 'liveAction': 'research_only_no_live_change'}
    write_json(OUT_VERDICT_JSON, verdict)
    lines = ['# top159 五模型概率一体化快搜', '', f'- 北京时间：`{payload["beijingTime"]}`', f'- 完成：`{len(results)}/{total}`', f'- 严格通过：`{len(strict)}`', '- live动作：`research_only_no_live_change`', '', '|候选|窗口|组合|模型|训练窗|交易数|胜/负|胜率|P5|P50|P95|最大回撤|月正收益|FAK+2资金|哈希|', '|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    def add(r):
        lines.append(f"|{r['name']}|{r['window']}|{r.get('combo','-')}|{r.get('engine','baseline')}|{r.get('trainWindow','-')}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']}|{r['endingBankrollP5']}|{r['endingBankrollP50']}|{r['endingBankrollP95']}|{r['maxDrawdownP50']}|{r['monthlyPositiveRatioP50']}|{r['fak2EndingBankroll']}|`{r['setHash']}`|")
    for r in [base_rows['180d'], base_rows['365d']]: add(r)
    for c in verdicts[:25]:
        for r in c['rows']: add(r)
    write_text(OUT_LEADERBOARD_MD, '\n'.join(lines)+'\n')
    vlines = ['# top159 五模型概率一体化快搜唯一结论', '', f'- 北京时间：`{payload["beijingTime"]}`', f'- 状态：`{verdict["status"]}`', '- live动作：`research_only_no_live_change`']
    if selected:
        vlines += [f'- 选中候选：`{selected["name"]}`', f'- 参数：`{json.dumps(selected["params"], ensure_ascii=False)}`']
    else:
        vlines += ['- 暂无过门候选；快搜仍在继续或需要扩大下一批。']
    write_text(OUT_VERDICT_MD, '\n'.join(vlines)+'\n')


def run() -> int:
    started=time.time()
    print(f'[fast] start {bj_now()} workers={WORKERS} maxSeconds={MAX_SECONDS}', flush=True)
    df, features, data_truth = S.build_stacked_frame()
    base_rows = S.base_rows()
    audit = S.bug_audit(df, features, base_rows, data_truth)
    data_truth = {**data_truth, 'fastMode': 'stage1_formula_all_combos_then_stage2_model_top_combos', 'bugAudit': audit}
    write_json(OUT_AUDIT_JSON, audit)
    print(f'[fast] frame rows={len(df)} features={len(features)} auditPassed={audit["passed"]}', flush=True)
    stage1 = stage1_candidates(df, base_rows)
    write_json(OUT_STAGE1_JSON, {'generatedAt': now_iso(), 'beijingTime': bj_now(), 'stage1Count': len(stage1), 'fullEvalCount': 0, 'topQuick': stage1[:300], 'topFullFormula': []})
    full_formula=[]
    for s in stage1[:FAST_FULL_EVAL_LIMIT]:
        try:
            full_formula.append(full_eval_formula(df, s))
        except Exception as exc:
            full_formula.append({'name': s['name'], 'params': s['params'], 'error': str(exc)[:300]})
    full_formula_valid=[x for x in full_formula if x.get('rows')]
    full_formula_valid.sort(key=lambda x: score_candidate(x, base_rows), reverse=True)
    write_json(OUT_STAGE1_JSON, {'generatedAt': now_iso(), 'beijingTime': bj_now(), 'stage1Count': len(stage1), 'fullEvalCount': len(full_formula), 'topQuick': stage1[:200], 'topFullFormula': full_formula_valid[:100]})
    model_params = derive_model_params(stage1)
    total = len(full_formula) + len(model_params)
    print(f'[fast] stage1={len(stage1)} fullFormula={len(full_formula)} modelCandidates={len(model_params)}', flush=True)
    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    results=list(full_formula)
    done_names={r.get('name') for r in results}
    # Do not reuse old file blindly; this fast run is intended to be fresh but resumable by name.
    if OUT_RESULTS.exists():
        for line in OUT_RESULTS.read_text(encoding='utf-8').splitlines():
            if not line.strip(): continue
            try:
                r=json.loads(line)
                if r.get('name') not in done_names:
                    results.append(r); done_names.add(r.get('name'))
            except Exception: pass
    pending=[]
    for i,p in enumerate(model_params):
        name=f"fast_model_{p['engine']}_{p['train_window']}_{'+'.join(p['combo'])}_{p['combo_mode']}_{p['base_mode']}_edge{p['edge']}_{S.stable_hash(p)}"
        if name not in done_names:
            pending.append((i,p))
    write_outputs(results, base_rows, total, started, data_truth, finished=False)
    last=time.time()
    with OUT_RESULTS.open('a', encoding='utf-8') as fh:
        # Write formula evals if not already present.
        for r in full_formula:
            if r.get('name') not in done_names:
                fh.write(json.dumps(r, ensure_ascii=False, default=str, separators=(',',':'))+'\n'); fh.flush(); done_names.add(r.get('name'))
        with cf.ProcessPoolExecutor(max_workers=WORKERS, initializer=init_worker, initargs=(df, features, base_rows)) as ex:
            futures={}
            it=iter(pending)
            for _ in range(max(WORKERS*2,4)):
                try: item=next(it)
                except StopIteration: break
                futures[ex.submit(eval_model_worker,item)]=item
            while futures:
                if time.time()-started > MAX_SECONDS: break
                done,_=cf.wait(futures,timeout=5,return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    item=futures.pop(fut)
                    try: row=fut.result()
                    except Exception as exc: row={'name':f'candidate_{item[0]}','params':item[1],'error':str(exc)[:500]}
                    fh.write(json.dumps(row, ensure_ascii=False, default=str, separators=(',',':'))+'\n'); fh.flush()
                    results.append(row)
                    try:
                        nxt=next(it); futures[ex.submit(eval_model_worker,nxt)]=nxt
                    except StopIteration: pass
                if time.time()-last >= CHECKPOINT_SECONDS:
                    write_outputs(results, base_rows, total, started, data_truth, finished=False)
                    print(f'[fast] checkpoint {len(results)}/{total}', flush=True)
                    last=time.time()
            for fut in futures: fut.cancel()
    finished = len([r for r in results if r.get('rows') or r.get('error')]) >= total
    write_outputs(results, base_rows, total, started, data_truth, finished=finished)
    print(json.dumps({'status':'finished' if finished else 'checkpointed','done':len(results),'total':total,'leaderboard':str(OUT_LEADERBOARD_MD),'verdict':str(OUT_VERDICT_MD)},ensure_ascii=False,indent=2),flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(run())
