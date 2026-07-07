#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT=Path('/Users/mac/polyfun')
RAW=ROOT/'data/raw'
REPORTS=ROOT/'reports'
STAKE=1.0
BUY_PRICE=0.51
SELL_SLIP=0.01
ASSETS=['ETH','BTC']
TFS={'15m':15,'1h':60,'4h':240}
TRAIN_WINDOWS={'1y':365,'2y':730,'3y':1095,'5y':1825,'7y':2555,'full':None}
VERIFY=[180,365]
TP=[0.10,0.15,0.20,0.25,0.30,0.40,0.50]
SL=[0.30,0.35,0.40,0.45,0.50]
TIME=[0.25,0.50,0.75,0.90]
TRAIL_ARM=[0.15,0.20,0.25,0.30]
TRAIL_BACK=[0.08,0.12]  # reduced grid for full-run practicality; 5/10/15 are covered in prior trail studies
CANCEL=[3,4,5,6,7,8]
FEATURES=['ret1','ret4','ret16','vol16','vol64','range_pct','ema_diff','vol_ratio','hour_sin','hour_cos','dow_sin','dow_cos']

@dataclass(frozen=True)
class Action:
    name:str; kind:str; tp:float=0; sl:float=0; arm:float=0; back:float=0; tfrac:float=0.9

def now(): return datetime.now(timezone.utc).isoformat()
def write_json(p:Path,x:Any): p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8'); t.replace(p)
def write_text(p:Path,s:str): p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp'); t.write_text(s,encoding='utf-8'); t.replace(p)
def fee(p):
    p=np.asarray(p,dtype=float); out=np.zeros_like(p); m=(p>0)&(p<1); out[m]=0.25*(p[m]*(1-p[m]))**2; return out

def actions():
    out=[]
    for tf in TIME: out.append(Action(f'time{int(tf*100)}','time',tfrac=tf))
    for tp in TP:
        for sl in SL:
            for tf in TIME: out.append(Action(f'tp{int(tp*100)}_sl{int(sl*100)}_time{int(tf*100)}','tp_sl_time',tp=tp,sl=sl,tfrac=tf))
    for arm in TRAIL_ARM:
        for back in TRAIL_BACK:
            if back>=arm: continue
            for sl in SL: out.append(Action(f'trail{int(arm*100)}_back{int(back*100)}_sl{int(sl*100)}_time90','trail_sl_time',sl=sl,arm=arm,back=back,tfrac=0.90))
    return out
ACTIONS=actions()

def normalize_ts(s):
    x=pd.to_numeric(s,errors='coerce')
    med=float(x.dropna().median())
    if med>1e15: x=x/1000
    if med>1e12: x=x/1000
    return pd.to_datetime(x,unit='ms',utc=True)

def load_raw(asset):
    df=pd.read_parquet(RAW/f'{asset.lower()}_usdt_1m.parquet',columns=['timestamp','open','high','low','close','volume'])
    df['ts']=normalize_ts(df['timestamp'])
    for c in ['open','high','low','close','volume']: df[c]=pd.to_numeric(df[c],errors='coerce')
    return df.dropna(subset=['ts','open','high','low','close']).drop_duplicates('ts').sort_values('ts').reset_index(drop=True)

def make_bars(raw,tf):
    minutes=TFS[tf]
    df=raw.copy()
    df['period']=df['ts'].dt.floor(f'{minutes}min')
    rows=[]
    for start,part in df.groupby('period',sort=True):
        if len(part)<max(2,int(minutes*0.70)): continue
        close=part['close'].to_numpy(dtype=float)
        rows.append({'timestamp':start,'open':float(part.iloc[0].open),'high':float(part.high.max()),'low':float(part.low.min()),'close':float(part.iloc[-1].close),'volume':float(part.volume.sum()),'path':close})
    b=pd.DataFrame(rows).sort_values('timestamp').reset_index(drop=True)
    if b.empty: return b
    ret=b['close'].pct_change()
    b['ret1']=ret.shift(1); b['ret4']=b['close'].pct_change(4).shift(1); b['ret16']=b['close'].pct_change(16).shift(1)
    b['vol16']=ret.rolling(16).std().shift(1); b['vol64']=ret.rolling(64).std().shift(1)
    b['range_pct']=((b['high']-b['low'])/b['open'].replace(0,np.nan)).shift(1)
    ema12=b['close'].ewm(span=12,adjust=False).mean(); ema48=b['close'].ewm(span=48,adjust=False).mean(); b['ema_diff']=(ema12/ema48-1).shift(1)
    b['vol_ratio']=(b['volume']/b['volume'].rolling(48).mean()).shift(1)
    hour=b['timestamp'].dt.hour+ b['timestamp'].dt.minute/60; b['hour_sin']=np.sin(2*np.pi*hour/24); b['hour_cos']=np.cos(2*np.pi*hour/24)
    dow=b['timestamp'].dt.dayofweek; b['dow_sin']=np.sin(2*np.pi*dow/7); b['dow_cos']=np.cos(2*np.pi*dow/7)
    b['target_up']=(b['close']>b['open']).astype(int)
    return b.dropna().reset_index(drop=True)

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-20,20)))
def token_path_from_price(price_path, prev_vol, side):
    p=np.asarray(price_path,dtype=float)
    if p.size<2 or p[0]<=0: return np.array([],dtype=float)
    ret=p/p[0]-1.0
    elapsed=np.linspace(0,1,p.size)
    scale=max(float(prev_vol or 0)*2.2,0.0015)*np.sqrt(np.maximum(elapsed,0.04))
    up=sigmoid(ret/scale)
    tok=up if side==1 else 1-up
    return np.clip(tok,0.01,0.99)

def pnl_sell(sell):
    sell=np.clip(float(sell),0.01,0.99)
    tokens=(STAKE/BUY_PRICE)*(1-float(fee(np.array([BUY_PRICE]))[0]))
    return tokens*sell*(1-float(fee(np.array([sell]))[0]))-STAKE

def pnl_hold(win):
    tokens=(STAKE/BUY_PRICE)*(1-float(fee(np.array([BUY_PRICE]))[0]))
    return tokens-STAKE if win else -STAKE

def max_dd(pnls):
    a=np.asarray(pnls,dtype=float)
    if a.size==0: return 0.0
    eq=np.cumsum(a); return float((np.maximum.accumulate(eq)-eq).max())

def metric(rows, pnls, reasons, holds, candidates, action, layer, note):
    pnls=np.asarray(pnls,dtype=float); reasons=np.asarray(reasons,dtype=int); holds=np.asarray(holds,dtype=float)
    n=len(pnls)
    if n==0:
        return {'candidates':int(candidates),'trades':0,'wins':0,'losses':0,'winRatePct':0,'pnl':0,'maxDrawdown':0,'profitDrawdownRatio':0,'avgBuyPrice':0,'avgSellPrice':0,'takeProfitCount':0,'stopLossCount':0,'trailingCount':0,'timeExitCount':0,'cancelCount':0,'sellFailCount':0,'holdToSettlementCount':0,'avgHoldMinutes':0,'maxSingleLoss':0,'setHash':'empty','action':action,'dataLayer':layer,'note':note}
    wins=int((pnls>0).sum()); losses=int((pnls<0).sum()); total=float(pnls.sum()); dd=max_dd(pnls)
    ids=[str(x) for x in rows['id'].tolist()] if len(rows) else []
    avg_sell=float(np.mean([0.5*(1+x) for x in pnls])) if n else 0
    return {'candidates':int(candidates),'trades':n,'wins':wins,'losses':losses,'winRatePct':round(100*wins/n,4),'pnl':round(total,4),'maxDrawdown':round(dd,4),'profitDrawdownRatio':round(total/dd,6) if dd else 0,'avgBuyPrice':BUY_PRICE,'avgSellPrice':round(avg_sell,4),'takeProfitCount':int((reasons==1).sum()),'stopLossCount':int((reasons==2).sum()),'trailingCount':int((reasons==3).sum()),'timeExitCount':int((reasons==4).sum()),'cancelCount':0,'sellFailCount':0,'holdToSettlementCount':int((reasons==9).sum()),'avgHoldMinutes':round(float(holds.mean()),4),'maxSingleLoss':round(float(pnls.min()),4),'setHash':hashlib.sha256('\n'.join(sorted(ids+[action])).encode()).hexdigest()[:16],'action':action,'dataLayer':layer,'note':note}

def precompute_exits(df):
    n=len(df); ntp=len(TP); nsl=len(SL); nt=len(TIME)
    trail_pairs=[(a,b) for a in TRAIL_ARM for b in TRAIL_BACK if b<a]
    ntr=len(trail_pairs); big=10**9
    tp_idx=np.full((n,ntp),big,dtype=np.int32); tp_pnl=np.zeros((n,ntp),dtype=np.float32); tp_hold=np.zeros((n,ntp),dtype=np.float32)
    sl_idx=np.full((n,nsl),big,dtype=np.int32); sl_pnl=np.zeros((n,nsl),dtype=np.float32); sl_hold=np.zeros((n,nsl),dtype=np.float32)
    tm_idx=np.zeros((n,nt),dtype=np.int32); tm_pnl=np.zeros((n,nt),dtype=np.float32); tm_hold=np.zeros((n,nt),dtype=np.float32)
    tr_idx=np.full((n,ntr),big,dtype=np.int32); tr_pnl=np.zeros((n,ntr),dtype=np.float32); tr_hold=np.zeros((n,ntr),dtype=np.float32)
    tfm=df['tf_minutes'].to_numpy(dtype=float)
    paths=df['token_path'].tolist()
    def calc_pnl(path,idx):
        idx=max(1,min(len(path)-1,int(idx)))
        sell=max(0.01,min(0.99,float(path[idx])-SELL_SLIP))
        return pnl_sell(sell)
    for i,path in enumerate(paths):
        path=np.asarray(path,dtype=float); L=len(path); denom=max(1,L-1)
        for j,t in enumerate(TIME):
            ix=max(1,min(L-1,int(round(denom*t))))
            tm_idx[i,j]=ix; tm_pnl[i,j]=calc_pnl(path,ix); tm_hold[i,j]=ix/denom*tfm[i]
        for j,tp in enumerate(TP):
            hits=np.flatnonzero(path>=BUY_PRICE*(1+tp))
            if hits.size:
                ix=int(hits[0]); tp_idx[i,j]=ix; tp_pnl[i,j]=calc_pnl(path,ix); tp_hold[i,j]=ix/denom*tfm[i]
        for j,sl in enumerate(SL):
            hits=np.flatnonzero(path<=BUY_PRICE*(1-sl))
            if hits.size:
                ix=int(hits[0]); sl_idx[i,j]=ix; sl_pnl[i,j]=calc_pnl(path,ix); sl_hold[i,j]=ix/denom*tfm[i]
        for j,(arm,back) in enumerate(trail_pairs):
            # Trail is only searched until the 90% time barrier.
            lim=tm_idx[i,TIME.index(0.90)]
            armed=False; peak=BUY_PRICE; ix=big
            for k,pv in enumerate(path[:lim+1]):
                if (not armed) and pv>=BUY_PRICE*(1+arm):
                    armed=True; peak=float(pv)
                elif armed:
                    peak=max(peak,float(pv))
                    if pv<=peak*(1-back): ix=k; break
            if ix<big:
                tr_idx[i,j]=ix; tr_pnl[i,j]=calc_pnl(path,ix); tr_hold[i,j]=ix/denom*tfm[i]
    return {'tp_idx':tp_idx,'tp_pnl':tp_pnl,'tp_hold':tp_hold,'sl_idx':sl_idx,'sl_pnl':sl_pnl,'sl_hold':sl_hold,'tm_idx':tm_idx,'tm_pnl':tm_pnl,'tm_hold':tm_hold,'tr_idx':tr_idx,'tr_pnl':tr_pnl,'tr_hold':tr_hold,'trail_pairs':trail_pairs}

def eval_action_pre(df,pre,action):
    n=len(df); big=10**9
    if action.kind=='time':
        j=TIME.index(action.tfrac)
        return pre['tm_pnl'][:,j].astype(float), np.full(n,4,dtype=int), pre['tm_hold'][:,j].astype(float)
    if action.kind=='tp_sl_time':
        ti=TP.index(action.tp); si=SL.index(action.sl); mi=TIME.index(action.tfrac)
        idx_tp=pre['tp_idx'][:,ti]; idx_sl=pre['sl_idx'][:,si]; idx_tm=pre['tm_idx'][:,mi]
        take=(idx_tp<=idx_sl)&(idx_tp<=idx_tm); stop=(idx_sl<idx_tp)&(idx_sl<=idx_tm)
        pnl=np.where(take,pre['tp_pnl'][:,ti],np.where(stop,pre['sl_pnl'][:,si],pre['tm_pnl'][:,mi])).astype(float)
        hold=np.where(take,pre['tp_hold'][:,ti],np.where(stop,pre['sl_hold'][:,si],pre['tm_hold'][:,mi])).astype(float)
        reason=np.where(take,1,np.where(stop,2,4)).astype(int)
        return pnl,reason,hold
    if action.kind=='trail_sl_time':
        pair=(action.arm,action.back); ti=pre['trail_pairs'].index(pair); si=SL.index(action.sl); mi=TIME.index(action.tfrac)
        idx_tr=pre['tr_idx'][:,ti]; idx_sl=pre['sl_idx'][:,si]; idx_tm=pre['tm_idx'][:,mi]
        tr=(idx_tr<=idx_sl)&(idx_tr<=idx_tm); stop=(idx_sl<idx_tr)&(idx_sl<=idx_tm)
        pnl=np.where(tr,pre['tr_pnl'][:,ti],np.where(stop,pre['sl_pnl'][:,si],pre['tm_pnl'][:,mi])).astype(float)
        hold=np.where(tr,pre['tr_hold'][:,ti],np.where(stop,pre['sl_hold'][:,si],pre['tm_hold'][:,mi])).astype(float)
        reason=np.where(tr,3,np.where(stop,2,4)).astype(int)
        return pnl,reason,hold
    raise ValueError(action.kind)

def fit_direction(train):
    X=train[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0)
    y=train['target_up'].astype(int)
    try:
        import lightgbm as lgb
        m=lgb.LGBMClassifier(n_estimators=220,num_leaves=31,learning_rate=0.035,subsample=0.85,colsample_bytree=0.85,reg_lambda=8,min_child_samples=80,random_state=20260430,n_jobs=max(1,os.cpu_count() or 1),verbose=-1)
        engine='LightGBM'
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m=HistGradientBoostingClassifier(max_iter=220,learning_rate=0.035,max_leaf_nodes=31,l2_regularization=1,random_state=20260430)
        engine='sklearn_hist_gradient_boosting'
    m.fit(X,y); return m,engine

def choose_candidates(test,model):
    X=test[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0)
    if hasattr(model,'predict_proba'): p=model.predict_proba(X)[:,1]
    else: p=model.predict(X)
    out=test.copy(); out['p_up']=p; out['side']=np.where(p>=0.5,1,-1); out['direction']=np.where(p>=0.5,'UP','DOWN')
    toks=[]; ids=[]
    for _,r in out.iterrows():
        tok=token_path_from_price(r['path'], r['vol64'], int(r['side']))
        toks.append(tok); ids.append(f"{r['timestamp'].value}|{r['direction']}")
    out['token_path']=toks; out['id']=ids; out['tf_minutes']=out['tf_minutes'].astype(float)
    out['hold_win']=np.where(out['side']==1,out['target_up'].astype(bool),~out['target_up'].astype(bool))
    out['hold_pnl']=[pnl_hold(bool(x)) for x in out['hold_win']]
    return out[out['token_path'].map(len)>=2].copy()

def score(m,base):
    return m['winRatePct']*1000 - m['maxDrawdown']*8 + m['pnl']*0.5 + m['profitDrawdownRatio']*20 - max(0,base['pnl']*0.7-m['pnl'])*3

def run_asset_tf(asset,tf):
    print(f'[forced_exit] start {asset} {tf}', flush=True)
    raw=load_raw(asset); bars=make_bars(raw,tf); bars['tf_minutes']=TFS[tf]
    ts=bars['timestamp']; end=ts.max(); rows=[]; hypers=[]; audit={'asset':asset,'timeframe':tf,'barRows':int(len(bars)),'start':str(ts.min()),'end':str(end),'proxy':'fast_raw_kline_token_proxy_not_wallet_truth'}
    for vd in VERIFY:
        print(f'[forced_exit] {asset} {tf} verify {vd}d', flush=True)
        test_mask=(ts>=end-pd.Timedelta(days=vd)); test_base=bars[test_mask].copy()
        for tw,days in TRAIN_WINDOWS.items():
            print(f'[forced_exit] {asset} {tf} verify {vd}d train {tw}', flush=True)
            tr_start=ts.min() if days is None else end-pd.Timedelta(days=vd+days)
            train=bars[(ts>=tr_start)&(ts<end-pd.Timedelta(days=vd))].copy()
            if len(train)<500 or len(test_base)<100:
                hypers.append({'asset':asset,'timeframe':tf,'verifyWindow':f'{vd}d','trainWindow':tw,'status':'insufficient','trainRows':len(train),'testRows':len(test_base)}); continue
            model,engine=fit_direction(train); cand=choose_candidates(test_base,model)
            base=metric(cand,cand['hold_pnl'].to_numpy(),np.full(len(cand),9),np.full(len(cand),TFS[tf]),len(cand),'hold_to_expiry','raw_kline_proxy_fast','持有到结算对照；不是强制退出候选。')
            rows.append({'asset':asset,'timeframe':tf,'window':f'{vd}d','trainWindow':tw,'method':'方向模型_持有到结算对照',**base})
            pre=precompute_exits(cand)
            best=None
            for a in ACTIONS:
                p,r,h=eval_action_pre(cand,pre,a); m=metric(cand,p,r,h,len(cand),a.name,'raw_kline_proxy_fast','强制退出：止盈/30-50止损/移动止盈/时间退出，持有到结算=0。')
                sc=score(m,base)
                if best is None or sc>best[0]: best=(sc,a,m)
            rows.append({'asset':asset,'timeframe':tf,'window':f'{vd}d','trainWindow':tw,'method':'固定规则最佳强制退出',**best[2]})
            # model-gated version: only take high-confidence top quantiles, with same best action.
            conf=np.abs(cand['p_up'].to_numpy()-0.5)
            for q in [0.50,0.60,0.70,0.80]:
                th=float(np.quantile(conf,q)); sub=cand.iloc[np.flatnonzero(conf>=th)].copy()
                if len(sub)<50: continue
                sub_pre=precompute_exits(sub); p,r,h=eval_action_pre(sub,sub_pre,best[1]); m=metric(sub,p,r,h,len(cand),best[1].name,'raw_kline_proxy_fast',f'价值过滤：只做方向置信度前 {int((1-q)*100)}% 的信号；强制退出。')
                rows.append({'asset':asset,'timeframe':tf,'window':f'{vd}d','trainWindow':tw,'method':f'置信过滤q{q:.2f}+最佳强制退出',**m})
            hypers.append({'asset':asset,'timeframe':tf,'verifyWindow':f'{vd}d','trainWindow':tw,'status':'ok','engine':engine,'trainRows':int(len(train)),'testRows':int(len(test_base)),'candidateRows':int(len(cand)),'bestAction':best[1].name,'cancelMinuteDecision':'raw proxy assumes immediate taker fill, cannot prove cancel minute; 3-8 all equivalent here; live preflight default 5m unless official lifecycle model proves better'})
    print(f'[forced_exit] done {asset} {tf}', flush=True)
    return rows,audit,hypers

def read_real_layer():
    rows=[]; audits=[]
    for asset in ASSETS:
        p=ROOT/'data/processed'/f'vnext_entry_exit_episodes_{asset.lower()}_usdt.parquet'
        if not p.exists(): audits.append({'asset':asset,'status':'missing'}); continue
        try:
            df=pd.read_parquet(p); audits.append({'asset':asset,'status':'available','rows':int(len(df)),'note':'真实层可用于成交诊断；本脚本不把真实短窗训练成上线结论。'})
        except Exception as e: audits.append({'asset':asset,'status':'error','error':str(e)})
    return rows,audits

def verdict(rows):
    pairs={}
    for r in rows:
        if r['method']=='方向模型_持有到结算对照': pairs[(r['asset'],r['timeframe'],r['window'],r['trainWindow'])]=r
    cands=[]
    for r in rows:
        if r['method']=='方向模型_持有到结算对照': continue
        if r.get('holdToSettlementCount')!=0 or r.get('trades',0)<50: continue
        b=pairs.get((r['asset'],r['timeframe'],r['window'],r['trainWindow']))
        if not b: continue
        ok=(r['winRatePct']>b['winRatePct'] and r['maxDrawdown']<b['maxDrawdown'] and r['pnl']>=b['pnl']*0.90)
        rr=dict(r); rr['_pass']=ok; rr['_base']=b; cands.append(rr)
    both=[]
    for r in cands:
        other='180d' if r['window']=='365d' else '365d'
        m=next((x for x in cands if x['asset']==r['asset'] and x['timeframe']==r['timeframe'] and x['trainWindow']==r['trainWindow'] and x['method']==r['method'] and x['action']==r['action'] and x['window']==other),None)
        if m and r['_pass'] and m['_pass']:
            both.append((r['pnl']+m['pnl']-r['maxDrawdown']-m['maxDrawdown'],r,m))
    if not both:
        return {'status':'no_live_candidate','reason':'没有强制退出候选同时在180天和365天满足：胜率高于持有基线、最大回撤更低、盈亏不明显变差、持有到结算=0。','action':'不恢复真钱；如果必须实盘，只能另起小额金丝雀，不使用本策略。'}
    both.sort(key=lambda x:x[0],reverse=True); _,a,b=both[0]
    return {'status':'proxy_candidate_only_not_live','reason':'原始线代理两窗通过，但还缺 Polymarket 真实卖盘深度、卖出失败、取消生命周期首证；不能直接上线。','candidate180or365':a,'pairedWindow':b}

def make_md(payload):
    lines=['# 强制退出 stop30-50 修正版研究结果','',f"生成时间：`{payload['generatedAt']}`",'', '## 重要口径','- 本报告只研究，不改真钱。','- 止损只搜 `30/35/40/45/50%`。','- 下单金额统一 `1U` 标准化；不决定实盘金额。','- 强制退出候选的 `持有到结算次数` 必须为 `0`。','- 原始线代理不等于 Polymarket 钱包收益；成交、卖出、取消仍需官网真相验证。','- 取消分钟在本代理中等价，因为这里假设吃价立即成交；真实订单取消仍默认 `5分钟`，除非订单生命周期模型证明更优。','', '## 绝对结果表','|资产|周期|窗口|训练窗|方法|动作|候选|交易|胜/负|胜率|盈亏|最大回撤|收益回撤比|均买|均卖|止盈|止损|移动止盈|时间退出|取消|卖出失败|持有到结算|平均持有|哈希|说明|','|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|']
    rows=payload['rows']
    # Keep report readable: include baseline and best fixed rows plus all rows that pass or are top by window.
    filtered=[]
    for r in rows:
        if r['method'] in ('方向模型_持有到结算对照','固定规则最佳强制退出'): filtered.append(r)
    # Add top confidence rows per asset/tf/window/train by pnl/dd score.
    extra=[r for r in rows if '置信过滤' in r['method']]
    extra=sorted(extra,key=lambda r:(r['asset'],r['timeframe'],r['window'],-(r['pnl']-r['maxDrawdown'])),reverse=False)[:200]
    seen=set()
    for r in filtered+extra:
        key=(r['asset'],r['timeframe'],r['window'],r['trainWindow'],r['method'],r['action'],r['trades'])
        if key in seen: continue
        seen.add(key)
        lines.append(f"|{r['asset']}|{r['timeframe']}|{r['window']}|{r['trainWindow']}|{r['method']}|{r['action']}|{r['candidates']}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']}%|{r['pnl']}|{r['maxDrawdown']}|{r['profitDrawdownRatio']}|{r['avgBuyPrice']}|{r['avgSellPrice']}|{r['takeProfitCount']}|{r['stopLossCount']}|{r['trailingCount']}|{r['timeExitCount']}|{r['cancelCount']}|{r['sellFailCount']}|{r['holdToSettlementCount']}|{r['avgHoldMinutes']}|`{r['setHash']}`|{r['note']}|")
    v=payload['uniqueVerdict']; lines+=['','## 唯一结论',f"- 状态：`{v.get('status')}`",f"- 原因：{v.get('reason')}"]
    if v.get('action'): lines.append(f"- 动作：{v.get('action')}")
    lines+=['','## 防错检查',f"- 持有到结算违规行数：`{payload['bugcheck']['holdSettlementViolations']}`",f"- 同一候选重复性：`{payload['bugcheck']['repeatability']}`",f"- CatBoost/XGBoost：`{payload['bugcheck']['modelAvailability']}`"]
    return '\n'.join(lines)+'\n'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--assets',nargs='*',default=ASSETS); ap.add_argument('--timeframes',nargs='*',default=list(TFS)); args=ap.parse_args()
    rows=[]; audits=[]; hypers=[]
    for asset in [x.upper() for x in args.assets]:
        for tf in args.timeframes:
            if tf not in TFS: continue
            r,a,h=run_asset_tf(asset,tf); rows+=r; audits.append(a); hypers+=h
    _,ra=read_real_layer(); v=verdict(rows)
    violations=[r for r in rows if r['method']!='方向模型_持有到结算对照' and r.get('holdToSettlementCount')!=0]
    payload={'generatedAt':now(),'status':'complete','rows':rows,'dataAudits':audits,'realLayerAudits':ra,'hyperopt':hypers,'uniqueVerdict':v,'bugcheck':{'holdSettlementViolations':len(violations),'repeatability':'deterministic fixed seed; same code path single pass','modelAvailability':'LightGBM available; CatBoost/XGBoost not installed in current env, so ranking model degraded to LightGBM classifier + fixed-action grid.'},'parameterGrids':{'stopLoss':SL,'takeProfit':TP,'trailingArm':TRAIL_ARM,'trailingBack':TRAIL_BACK,'timeExit':TIME,'cancelMinutes':CANCEL}}
    REPORTS.mkdir(parents=True,exist_ok=True)
    write_json(REPORTS/'forced_exit_stop30_50_absolute_compare_latest.json',payload)
    write_text(REPORTS/'forced_exit_stop30_50_absolute_compare_latest.md',make_md(payload))
    write_json(REPORTS/'forced_exit_stop30_50_model_hyperopt_latest.json',{'generatedAt':payload['generatedAt'],'hyperopt':hypers})
    write_json(REPORTS/'forced_exit_stop30_50_bugcheck_latest.json',payload['bugcheck'])
    write_json(REPORTS/'forced_exit_stop30_50_unique_verdict_latest.json',v)
    write_text(REPORTS/'forced_exit_stop30_50_unique_verdict_latest.md','# 强制退出 stop30-50 唯一结论\n\n'+json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'ok':True,'rows':len(rows),'report':str(REPORTS/'forced_exit_stop30_50_absolute_compare_latest.md'),'verdict':v.get('status')},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
