#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path('/Users/mac/polyfun')
REPORTS = ROOT / 'reports'
SHOCK_SCRIPT = ROOT / 'polyfun-next' / 'scripts' / 'run_top159_shock_candle_filter_research.py'
BASE_SCRIPT = ROOT / 'scripts' / 'ops' / 'run_crypto15m_1h_multasset_pressure_search_latest.py'
ARCHIVE_SCRIPT = ROOT / 'scripts' / 'ops' / 'run_top159_all_archived_real_eth_compare.py'
OUT_JSON = REPORTS / 'top159_shock_filter_pnl_drawdown_compare_latest.json'
OUT_MD = REPORTS / 'top159_shock_filter_pnl_drawdown_compare_latest.md'
START_BANKROLL = 850.0
STAKE_PCT = 0.01
SELECTED_SHOCK_PARAMS = {
    'family': 'model_gate',
    'action': 'hard_block',
    'rule_mode': 'any_all',
    'body_min': 0.45,
    'range_q_min': 0.60,
    'volume_mult_min': 0.0,
    'model_engine': 'lightgbm',
    'min_gate_win_prob': 0.54,
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

shock = load_module('shock_filter_compare_mod', SHOCK_SCRIPT)
base = load_module('crypto_pressure_compare_mod', BASE_SCRIPT)
archive = load_module('archive_real_compare_mod', ARCHIVE_SCRIPT)
base.TRIALS = 1000


def max_drawdown_from_pnl(pnls: list[float]) -> float:
    if not pnls:
        return 0.0
    curve = np.concatenate([[0.0], np.cumsum(np.asarray(pnls, dtype=float))])
    peak = np.maximum.accumulate(curve)
    return round(float(np.max(peak - curve)), 6)


def summarize_compound(df: pd.DataFrame, name: str, window: str, method: str, entry_price: float, fill: np.ndarray | None = None) -> dict[str, Any]:
    d = df.sort_values('dt').reset_index(drop=True)
    won = d['won'].astype(bool).to_numpy()
    returns = base.returns_from_wins(won, entry_price)
    if fill is None:
        fill = np.ones(len(won), dtype=float)
    row = base.summarize_curve(pd.to_datetime(d['dt'], utc=True), won, returns, fill, method, 'ETH', '15m', window, name, {'entryPrice': entry_price})
    return {
        'name': name,
        'window': window,
        'method': method,
        'trades': int(len(d)),
        'wins': int(won.sum()),
        'losses': int((~won).sum()),
        'winRatePct': row['signalWinRatePct'],
        'endingBankroll': row['endingBankroll'],
        'compoundPnl': row['compoundPnl'],
        'maxDrawdownUsd': row['maxDrawdownUsd'],
        'maxDrawdownPct': row['maxDrawdownPct'],
        'returnDrawdownRatio': row['returnDrawdownRatio'],
        'monthlyPositiveRatio': row['monthlyPositiveRatio'],
        'drawdownPeakTime': row['drawdownPeakTime'],
        'drawdownTroughTime': row['drawdownTroughTime'],
        'setHash': row['setHash'],
    }


def summarize_toxic_mc(df: pd.DataFrame, name: str, window: str) -> dict[str, Any]:
    d = df.sort_values('dt').reset_index(drop=True)
    won = d['won'].astype(bool).to_numpy()
    returns = base.returns_from_wins(won, 0.50)
    mc = base.monte_carlo_slot1(pd.to_datetime(d['dt'], utc=True), won, returns, 20260503 + (180 if window == '180d' else 365) + len(d))
    return {
        'name': name,
        'window': window,
        'method': 'slot1_toxic_monte_carlo',
        'trades': int(len(d)),
        'wins': int(won.sum()),
        'losses': int((~won).sum()),
        'winRatePct': round(100.0 * float(won.mean()), 6) if len(won) else 0.0,
        **mc,
        'winnerFillRatePct': round(base.SLOT1_WIN_FILL * 100, 6),
        'loserFillRatePct': round(base.SLOT1_LOSS_FILL * 100, 6),
    }


def build_validation_sets() -> dict[str, dict[str, pd.DataFrame]]:
    selected, _truth = shock.build_selected_sets()
    enriched = shock.enrich_shock_features(selected)
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for window in ['180d', '365d']:
        train = enriched[enriched['period_name'] == f'gate_train_for_{window}'].copy()
        val = enriched[enriched['period_name'] == f'validation_{window}'].copy()
        model, cols = shock.fit_gate_model(train, SELECTED_SHOCK_PARAMS['model_engine'])
        prob = shock.predict_gate(model, cols, val)
        cond = shock.condition_mask(val, SELECTED_SHOCK_PARAMS)
        keep = (~cond) | (prob >= float(SELECTED_SHOCK_PARAMS['min_gate_win_prob']))
        base_df = val.sort_values('dt').reset_index(drop=True)
        shock_df = val[keep].copy().sort_values('dt').reset_index(drop=True)
        out[window] = {'current_archive_top159': base_df, 'shock_filter': shock_df, 'all_val_with_keep': val.assign(shock_keep=keep.to_numpy(), shock_prob=prob, shock_condition=cond.to_numpy())}
    return out


def validation_rows(compare_sets: dict[str, dict[str, pd.DataFrame]]) -> list[dict[str, Any]]:
    rows = []
    for window, sets in compare_sets.items():
        for name, df in [('current_new_archive_top159', sets['current_archive_top159']), ('shock_filter_gate', sets['shock_filter'])]:
            rows.append(summarize_compound(df, name, window, 'full_fill_buy_0.50', 0.50))
            rows.append(summarize_compound(df, name, window, 'fak_pressure_buy_0.52', 0.52))
            rows.append(summarize_toxic_mc(df, name, window))
    return rows


def real_archive_scope_rows(old: pd.DataFrame, val_with_keep: pd.DataFrame, scope_name: str) -> list[dict[str, Any]]:
    # Compare current selected rows and shock-kept selected rows only on old archived real market timestamps.
    pred = val_with_keep[['dt','pred_up15','score15','won','shock_keep','shock_prob','shock_condition']].copy()
    merged = old.merge(pred, left_on='marketStart', right_on='dt', how='left')
    rows = []
    for name, mask in [
        ('current_new_archive_top159_on_archived_markets', merged['pred_up15'].notna()),
        ('shock_filter_gate_on_archived_markets', merged['shock_keep'].fillna(False).astype(bool)),
    ]:
        s = merged[mask].copy().sort_values('marketStart')
        if len(s):
            model_won = s['pred_up15'].astype(bool).to_numpy() == s['actualUp'].astype(bool).to_numpy()
            one = np.where(model_won, 1.0, -1.0)
            fak52 = np.where(model_won, (1.0 / 0.52) - 1.0, -1.0)
            s['sameDirectionAsOld'] = s['pred_up15'].astype(bool).to_numpy() == (s['direction'].astype(str).str.upper() == 'UP').to_numpy()
            same = s[s['sameDirectionAsOld']].copy()
            same_pnl = same['pnl'].astype(float).tolist()
            wins = int(model_won.sum()); losses = int(len(model_won) - wins)
            blocked_mask = ~old['marketStart'].isin(s['marketStart'])
            blocked = old[blocked_mask]
            rows.append({
                'scope': scope_name,
                'name': name,
                'oldRealMarkets': int(len(old)),
                'selectedTrades': int(len(s)),
                'skippedTrades': int(len(old) - len(s)),
                'skippedOldWinners': int(blocked['won'].sum()) if len(blocked) else 0,
                'skippedOldLosers': int((~blocked['won']).sum()) if len(blocked) else 0,
                'wins': wins,
                'losses': losses,
                'winRatePct': round(100.0 * wins / len(s), 6) if len(s) else 0.0,
                'oneUnitPnl': round(float(one.sum()), 6),
                'oneUnitMaxDrawdown': max_drawdown_from_pnl(one.tolist()),
                'fak52Pnl': round(float(fak52.sum()), 6),
                'fak52MaxDrawdown': max_drawdown_from_pnl(fak52.tolist()),
                'sameDirectionExecutableTrades': int(len(same)),
                'sameDirectionActualPnlUsd': round(float(sum(same_pnl)), 6) if same_pnl else 0.0,
                'sameDirectionActualMaxDrawdownUsd': max_drawdown_from_pnl(same_pnl),
                'avgScore': round(float(s['score15'].mean()), 6),
                'shockConditionTrades': int(s['shock_condition'].fillna(False).astype(bool).sum()),
                'avgShockProb': round(float(s['shock_prob'].dropna().mean()), 6) if s['shock_prob'].notna().any() else 0.0,
                'setHash': archive.stable_hash(s[['marketSlug','pred_up15','actualUp']].to_dict('records')),
            })
        else:
            rows.append({'scope': scope_name, 'name': name, 'oldRealMarkets': int(len(old)), 'selectedTrades': 0})
    return rows


def archived_real_rows(compare_sets: dict[str, dict[str, pd.DataFrame]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fill_rows, scan_audit = archive.load_live_fill_rows()
    all_eth = archive.aggregate_logical(fill_rows, '全部归档ETH15m live去重')
    strict_mask = fill_rows['sourcePath'].str.contains('slot01|forcedtrade|base_forcedtrade|__live_eth|logs_v5_exp10_bp0480__live_eth', case=False, regex=True)
    strict = archive.aggregate_logical(fill_rows[strict_mask].copy(), '严格旧slot1/ETH相关live去重') if strict_mask.any() else all_eth.iloc[0:0]
    val180 = compare_sets['180d']['all_val_with_keep']
    rows: list[dict[str, Any]] = []
    if len(strict):
        rows += real_archive_scope_rows(strict, val180, '严格旧slot1/ETH相关live去重')
    rows += real_archive_scope_rows(all_eth, val180, '全部归档ETH15m live去重')
    audit = {
        'scanAudit': scan_audit,
        'strictTrades': int(len(strict)),
        'allArchivedTrades': int(len(all_eth)),
        'archivedRangeUtc': [str(all_eth['marketStart'].min()), str(all_eth['marketStart'].max())],
        'truthLimit': 'new/shock models were not actually executed in old archived windows; sameDirectionActualPnlUsd only reuses old fills when model direction equals old executed direction',
    }
    return rows, audit


def md_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    lines = ['|' + '|'.join(headers) + '|', '|' + '|'.join(['---'] * len(headers)) + '|']
    for r in rows:
        lines.append('|' + '|'.join(str(r.get(h, '')) for h in headers) + '|')
    return '\n'.join(lines)


def render_md(payload: dict[str, Any]) -> str:
    lines = ['# top159 冲击K线过滤：盈亏与回撤对比', '', f"- 北京时间：`{pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y-%m-%d %H:%M:%S CST')}`", '- live动作：`research_only_no_live_change`', '- 口径说明：180/365 是模型回测资金曲线；历史真实交易是旧真实市场窗口上的方向/同向成交复用，不等同于新模型真钱实盘。', '']
    lines += ['## 180天 / 365天资金曲线', '']
    headers = ['name','window','method','trades','wins','losses','winRatePct','endingBankroll','compoundPnl','maxDrawdownUsd','maxDrawdownPct','returnDrawdownRatio','monthlyPositiveRatio']
    rows = [r for r in payload['validationRows'] if r['method'] != 'slot1_toxic_monte_carlo']
    lines.append(md_table(rows, headers))
    lines += ['', '## slot1成交毒性蒙特卡洛', '']
    headers2 = ['name','window','trades','wins','losses','winRatePct','endingBankrollP5','endingBankrollP50','endingBankrollP95','compoundPnlP50','maxDrawdownP50','maxDrawdownP95','monthlyPositiveRatioP50','winnerFillRatePct','loserFillRatePct']
    rows2 = [r for r in payload['validationRows'] if r['method'] == 'slot1_toxic_monte_carlo']
    lines.append(md_table(rows2, headers2))
    lines += ['', '## 历史全部真实交易窗口', '']
    headers3 = ['scope','name','oldRealMarkets','selectedTrades','skippedTrades','wins','losses','winRatePct','oneUnitPnl','oneUnitMaxDrawdown','fak52Pnl','fak52MaxDrawdown','sameDirectionExecutableTrades','sameDirectionActualPnlUsd','sameDirectionActualMaxDrawdownUsd','shockConditionTrades','avgScore','avgShockProb']
    lines.append(md_table(payload['archivedRealRows'], headers3))
    lines += ['', '## 审计', '```json', json.dumps(payload['audit'], ensure_ascii=False, indent=2, default=str), '```']
    return '\n'.join(lines) + '\n'


def main() -> int:
    sets = build_validation_sets()
    val_rows = validation_rows(sets)
    arch_rows, arch_audit = archived_real_rows(sets)
    payload = {
        'generatedAt': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
        'selectedShockParams': SELECTED_SHOCK_PARAMS,
        'validationRows': val_rows,
        'archivedRealRows': arch_rows,
        'audit': {
            'liveAction': 'research_only_no_live_change',
            'validationTruth': 'same validation rows as shock filter research; current_new_archive_top159 means live profile replayed per validation window',
            'bankroll': {'start': START_BANKROLL, 'stakePct': STAKE_PCT},
            'methods': {
                'full_fill_buy_0.50': 'proxy buy price 0.50, full fill, compounding 1%',
                'fak_pressure_buy_0.52': 'proxy buy price 0.52, full fill, compounding 1%',
                'slot1_toxic_monte_carlo': 'winner fill 74.4728%, loser fill 92.1412%, 1000 trials',
                'archived_same_direction_actual': 'only old actual fills whose executed direction matches model direction can reuse old realized pnl',
            },
            'archiveAudit': arch_audit,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    OUT_MD.write_text(render_md(payload), encoding='utf-8')
    print(json.dumps({'ok': True, 'md': str(OUT_MD), 'json': str(OUT_JSON)}, ensure_ascii=False, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
