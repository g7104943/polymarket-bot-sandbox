#!/usr/bin/env python3
from __future__ import annotations

# Research-only wide-fast random hyperopt for top159 shock filter.
# No live config mutation, no order submission.

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault('OMP_NUM_THREADS', os.environ.get('TOP159_SHOCK_WIDE_THREADS', '2'))
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', os.environ.get('TOP159_SHOCK_WIDE_THREADS', '2'))
os.environ.setdefault('MKL_NUM_THREADS', os.environ.get('TOP159_SHOCK_WIDE_THREADS', '2'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', os.environ.get('TOP159_SHOCK_WIDE_THREADS', '2'))

import numpy as np
import pandas as pd

ROOT = Path('/Users/mac/polyfun')
REPORTS = ROOT / 'reports'
EXTREME_SCRIPT = ROOT / 'polyfun-next' / 'scripts' / 'run_top159_shock_filter_extreme_search.py'
OUT_JSON = REPORTS / 'top159_shock_filter_wide_fast_search_latest.json'
OUT_MD = REPORTS / 'top159_shock_filter_wide_fast_verdict_latest.md'
OUT_CAND = REPORTS / 'top159_shock_filter_wide_fast_candidates_latest.jsonl'
RNG_SEED = 20260503


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

ext = load_module('shock_filter_wide_extreme', EXTREME_SCRIPT)
shock = ext.shock
base = ext.base
base.TRIALS = int(os.environ.get('TOP159_SHOCK_WIDE_MC_TRIALS', '1000'))


def bj_now() -> str:
    return pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y-%m-%d %H:%M:%S CST')


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def fit_gate_model_wide(train: pd.DataFrame, hyper: dict[str, Any]):
    x, cols = shock.model_features(train)
    y = train['won'].astype(int).to_numpy()
    if len(np.unique(y)) < 2 or len(train) < 200:
        return None, cols
    engine = hyper['engine']
    if engine in ('lightgbm', 'logistic'):
        return ext.fit_gate_model_ext(train, hyper)
    if engine == 'xgboost':
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=int(hyper['n_estimators']), max_depth=int(hyper['max_depth']), learning_rate=float(hyper['learning_rate']),
            subsample=float(hyper['subsample']), colsample_bytree=float(hyper['colsample_bytree']), reg_lambda=float(hyper['reg_lambda']),
            min_child_weight=float(hyper['min_child_weight']), objective='binary:logistic', eval_metric='logloss', tree_method='hist',
            random_state=RNG_SEED, n_jobs=2,
        )
    elif engine == 'catboost':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=int(hyper['iterations']), depth=int(hyper['depth']), learning_rate=float(hyper['learning_rate']),
            l2_leaf_reg=float(hyper['l2_leaf_reg']), loss_function='Logloss', random_seed=RNG_SEED,
            thread_count=2, verbose=False, allow_writing_files=False,
        )
    else:
        raise ValueError(engine)
    model.fit(x, y)
    return model, cols


def predict_gate_any(model: Any, cols: list[str], val: pd.DataFrame) -> np.ndarray:
    x, _ = shock.model_features(val)
    for c in cols:
        if c not in x.columns:
            x[c] = 0
    x = x[cols]
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def hyper_groups() -> list[dict[str, Any]]:
    return [
        {'engine':'logistic','C':0.2}, {'engine':'logistic','C':0.7}, {'engine':'logistic','C':2.0},
        {'engine':'lightgbm','n_estimators':80,'learning_rate':0.03,'num_leaves':12,'min_child_samples':35,'reg_lambda':0.7},
        {'engine':'lightgbm','n_estimators':110,'learning_rate':0.035,'num_leaves':18,'min_child_samples':55,'reg_lambda':1.2},
        {'engine':'lightgbm','n_estimators':160,'learning_rate':0.025,'num_leaves':28,'min_child_samples':90,'reg_lambda':2.2},
        {'engine':'xgboost','n_estimators':90,'max_depth':2,'learning_rate':0.04,'subsample':0.9,'colsample_bytree':0.85,'reg_lambda':1.0,'min_child_weight':3},
        {'engine':'xgboost','n_estimators':130,'max_depth':3,'learning_rate':0.035,'subsample':0.85,'colsample_bytree':0.9,'reg_lambda':1.6,'min_child_weight':6},
        {'engine':'xgboost','n_estimators':180,'max_depth':2,'learning_rate':0.025,'subsample':0.95,'colsample_bytree':0.75,'reg_lambda':2.5,'min_child_weight':10},
        {'engine':'catboost','iterations':90,'depth':3,'learning_rate':0.045,'l2_leaf_reg':3.0},
        {'engine':'catboost','iterations':130,'depth':4,'learning_rate':0.035,'l2_leaf_reg':5.0},
        {'engine':'catboost','iterations':170,'depth':3,'learning_rate':0.025,'l2_leaf_reg':8.0},
    ]


def random_conditions(n: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(RNG_SEED)
    modes = ext.RULE_MODES
    out = []
    for _ in range(n):
        out.append({
            'rule_mode': str(rng.choice(modes)),
            'body_min': round(float(rng.uniform(0.25, 0.75)), 3),
            'range_q_min': round(float(rng.uniform(0.35, 0.95)), 3),
            'volume_mult_min': float(rng.choice([0.0, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5])),
        })
    # Include around known good regions deterministically.
    for mode in ['any_all','1h_all','4h_all','any_terminal','4h_no_confirm','both_1h_4h']:
        for b in [0.30,0.325,0.35,0.375,0.40,0.425,0.45,0.475,0.50]:
            for rq in [0.40,0.45,0.50,0.55,0.60,0.65]:
                out.append({'rule_mode':mode,'body_min':b,'range_q_min':rq,'volume_mult_min':0.0})
    # de-dupe stable
    seen=set(); uniq=[]
    for c in out:
        h=ext.stable_hash(c)
        if h not in seen:
            seen.add(h); uniq.append(c)
    return uniq


def evaluate_rule_random(enriched: pd.DataFrame, conds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows=[]
    for i,c in enumerate(conds,1):
        p={'family':'rule','action':'hard_block',**c}
        rows.extend(ext.evaluate_one_params(enriched,p))
        for ss in [0.54,0.56,0.58,0.60,0.62,0.64]:
            p2={'family':'rule','action':'raise_score','shock_score_min':ss,**c}
            rows.extend(ext.evaluate_one_params(enriched,p2))
        if i%1000==0:
            print(f'[wide] rule conds={i}/{len(conds)}', flush=True)
    return rows


def evaluate_model_random(enriched: pd.DataFrame, conds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows=[]
    min_probs=[round(x,3) for x in np.arange(0.45,0.6501,0.01)]
    for hyper in hyper_groups():
        print(f'[wide] model {hyper}', flush=True)
        for window in ['180d','365d']:
            train=enriched[enriched['period_name']==f'gate_train_for_{window}'].copy()
            val=enriched[enriched['period_name']==f'validation_{window}'].copy()
            model,cols=fit_gate_model_wide(train,hyper)
            if model is None:
                continue
            prob=predict_gate_any(model,cols,val)
            for c in conds:
                cond=shock.condition_mask(val,c).to_numpy(dtype=bool)
                if int(cond.sum())<25:
                    continue
                for mp in min_probs:
                    p={'family':'model_gate','action':'hard_block',**c,'model_engine':hyper['engine'],'model_hyper':hyper,'min_gate_win_prob':mp}
                    keep=(~cond)|(prob>=mp)
                    rows.append(ext.fast_metric(val, keep, cond, ext.candidate_name(p), p, window))
    return rows


def main() -> int:
    t0=time.time()
    print(f'[wide] start {bj_now()}', flush=True)
    enriched, truth = ext.load_or_build_enriched()
    audit=ext.bug_audit(enriched)
    print(f'[wide] rows={len(enriched)} auditPassed={audit["passed"]}', flush=True)
    n=int(os.environ.get('TOP159_SHOCK_WIDE_RANDOM_CONDITIONS','2500'))
    conds=random_conditions(n)
    print(f'[wide] conditions={len(conds)}', flush=True)
    rows=[]
    rows.extend(ext.evaluate_baseline(enriched))
    rows.extend(ext.evaluate_one_params(enriched, ext.CURRENT_SHOCK_PARAMS))
    rows.extend(evaluate_rule_random(enriched, conds))
    rows.extend(evaluate_model_random(enriched, conds))
    board=ext.group_rows(rows)
    strict=[c for c in board if c['passedFast'] and c['name']!='current_new_archive_top159']
    selected=strict[0] if strict else (board[0] if board else None)
    with OUT_CAND.open('w',encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r,ensure_ascii=False,sort_keys=True,default=str)+'\n')
    named=[('current_new_archive_top159', {'family':'baseline'}), ('current_shock_filter_gate', ext.CURRENT_SHOCK_PARAMS)]
    if selected:
        named.append(('wide_fast_best', selected['params']))
    val,_maps=ext.final_validation_rows(enriched,named)
    arch,arch_audit=ext.archived_real_rows(enriched,named)
    payload={
        'generatedAt': pd.Timestamp.utcnow().isoformat(),
        'beijingTime': bj_now(),
        'liveAction': 'research_only_no_live_change',
        'runtimeSeconds': round(time.time()-t0,3),
        'dataTruth': truth,
        'bugAudit': audit,
        'conditions': len(conds),
        'rowCount': len(rows),
        'candidateGroups': len(board),
        'strictPassCount': len(strict),
        'selected': selected,
        'leaderboard': board[:200],
        'validationRows': val,
        'archivedRealRows': arch,
        'archiveAudit': arch_audit,
    }
    write_json(OUT_JSON,payload)
    lines=['# top159 冲击过滤宽搜快跑结论','',f'- 北京时间：`{payload["beijingTime"]}`',f'- live动作：`{payload["liveAction"]}`',f'- 审计通过：`{audit["passed"]}`',f'- 运行秒数：`{payload["runtimeSeconds"]}`',f'- 候选组：`{payload["candidateGroups"]}`',f'- 严格过门：`{payload["strictPassCount"]}`','']
    if selected:
        lines += ['## 最强宽搜候选',f"- `{selected['name']}`",'```json',json.dumps(selected['params'],ensure_ascii=False,indent=2,default=str),'```']
    lines += ['','## 文件',f'- JSON：`{OUT_JSON}`',f'- 候选明细：`{OUT_CAND}`']
    OUT_MD.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'ok':True,'selected':selected['name'] if selected else None,'runtimeSeconds':payload['runtimeSeconds'],'json':str(OUT_JSON),'md':str(OUT_MD)},ensure_ascii=False,indent=2),flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
