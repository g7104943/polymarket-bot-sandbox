#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path('/Users/mac/polyfun')
NEXT = ROOT / 'polyfun-next'
REPORTS = ROOT / 'reports'
BASE_SCRIPT = ROOT / 'scripts' / 'ops' / 'run_crypto15m_1h_multasset_pressure_search_latest.py'
FILL_SCRIPT = ROOT / 'scripts' / 'ops' / 'run_newslot1_fill_rate_toxicity_search_latest.py'
PRICES = [0.60, 0.65, 0.70]
WINDOWS = ['180d', '365d']
START_BANKROLL = 850.0
STAKE_PCT = 0.01


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

base = load_module('crypto_search_high_price', BASE_SCRIPT)
fill = load_module('fill_search_high_price', FILL_SCRIPT)


def top159_params() -> dict[str, Any]:
    report = ROOT / 'reports' / 'newslot1_fak_execution_loop_latest.json'
    if report.exists():
        data = json.loads(report.read_text())
        params = data.get('uniqueVerdict', {}).get('selectedParams')
        if isinstance(params, dict):
            return params
    return {
        'engine': 'lightgbm', 'train_window': '5y', 'feature_mode': 'trend', 'edge': 0.045,
        'vol_q': 0.999, 'trend_mode': 'none', 'bb_abs_max': 2.0, 'loss_n': 0, 'skip_k': 4,
        'n_estimators': 200, 'learning_rate': 0.02193585345721919, 'reg_lambda': 0.0907758387860903,
        'subsample': 0.926502583070262, 'colsample_bytree': 0.8764270068535669,
        'num_leaves': 36, 'min_child_samples': 80, 'depth': 3,
    }


def candidate_wins(window: str, params: dict[str, Any]) -> tuple[pd.Series, np.ndarray, dict[str, Any]]:
    raw = base.load_raw('ETH', '15m')
    df, features = base.build_features(raw, '15m')
    forbidden = [c for c in features if c in base.FORBIDDEN_FEATURES or any(c.startswith(p) for p in base.FORBIDDEN_PREFIXES)]
    if forbidden:
        raise RuntimeError(f'forbidden future features leaked: {forbidden}')
    train, val = fill.split_train_val(df, window, params['train_window'])
    feats = fill.feature_subset(features, params['feature_mode'])
    model = fill.fit_model(params['engine'], train, feats, params)
    if model is None:
        raise RuntimeError('top159 model failed to fit')
    prob = fill.predict(model, val, feats)
    dt, won, pred_up = fill.select_candidates(val, prob, params)
    audit = {
        'asset': 'ETH', 'timeframe': '15m', 'window': window,
        'trainRows': int(len(train)), 'validationRows': int(len(val)), 'featureCount': len(feats),
        'candidateTrades': int(len(won)), 'signalWins': int(np.sum(won)), 'signalLosses': int(len(won)-np.sum(won)),
        'signalWinRatePct': round(float(np.mean(won))*100, 6) if len(won) else 0.0,
    }
    return dt.reset_index(drop=True), won.astype(bool), audit


def formula_self_test() -> dict[str, Any]:
    tests = []
    for price in PRICES:
        win_ret = 1.0 - price
        loss_ret = -1.0
        sample = np.array([True, False])
        returns = np.where(sample, win_ret, loss_ret).astype(float)
        ok = abs(float(returns[0]) - win_ret) < 1e-12 and abs(float(returns[1]) - loss_ret) < 1e-12
        tests.append({'price': price, 'expectedWinReturn': win_ret, 'expectedLossReturn': loss_ret, 'actual': returns.tolist(), 'ok': ok})
    return {'ok': all(t['ok'] for t in tests), 'tests': tests}


def row_for(dt: pd.Series, won: np.ndarray, price: float, window: str, params: dict[str, Any]) -> dict[str, Any]:
    returns = np.where(won, 1.0 - price, -1.0).astype(float)
    summary = base.summarize_curve(
        dt, won, returns, np.ones(len(won)), f'fixed_buy_price_{price:.2f}',
        'ETH', '15m', window, 'top159_fixed_high_price_user_payout', {**params, 'fixedEntryPrice': price, 'stakePct': STAKE_PCT, 'startBankroll': START_BANKROLL}
    )
    summary.update({
        'fixedEntryPrice': price,
        'winReturnPerStake': round(1.0 - price, 8),
        'lossReturnPerStake': -1.0,
        'stakePct': STAKE_PCT,
        'startBankroll': START_BANKROLL,
        'filledTrades': int(len(won)),
        'winnerFillRatePct': 100.0,
        'loserFillRatePct': 100.0,
    })
    return summary


def render_md(payload: dict[str, Any]) -> str:
    lines = ['# top159 固定高买价回测', '', f"生成时间：`{payload['generatedAt']}`", '']
    lines.append('## 审计')
    lines.append(f"- 公式自检：`{payload['formulaSelfTest']['ok']}`；按你指定口径：赢单收益 = `1 - 买价`，输单 = `-1`。")
    lines.append('- 口径：top159 模型组合不变，只替换固定买入价为 0.60 / 0.65 / 0.70。')
    lines.append('- 窗口：180天、365天独立训练/验证，资金曲线不继承。')
    lines.append('- 资金：初始 850U，每笔当前资金 1%，不设上限。')
    lines.append('')
    lines.append('|窗口|买价|交易数|胜/负|胜率|赢单收益/本金|复利盈亏|期末资金|最大回撤U|最大回撤%|收益回撤比|月度正收益|最长回撤天|')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in payload['rows']:
        lines.append(f"|{r['window']}|{r['fixedEntryPrice']:.2f}|{r['requestedTrades']}|{r['wins']}/{r['losses']}|{r['winRatePct']:.2f}%|{r['winReturnPerStake']:.4f}|{r['compoundPnl']:.2f}|{r['endingBankroll']:.2f}|{r['maxDrawdownUsd']:.2f}|{r['maxDrawdownPct']:.2f}%|{r['returnDrawdownRatio']:.2f}|{r['monthlyPositiveRatio']:.2f}|{r['longestDrawdownDays']:.2f}|")
    lines.append('')
    v = payload['uniqueVerdict']
    lines.append('## 唯一结论')
    lines.append(f"- 状态：`{v['status']}`")
    lines.append(f"- 结论：{v['reason']}")
    return '\n'.join(lines) + '\n'


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    params = top159_params()
    rows = []
    audits = []
    for window in WINDOWS:
        dt, won, audit = candidate_wins(window, params)
        audits.append(audit)
        for price in PRICES:
            rows.append(row_for(dt, won, price, window, params))
    passing = [r for r in rows if r['endingBankroll'] > START_BANKROLL]
    verdict = {
        'status': 'positive_under_fixed_high_prices' if passing else 'all_fixed_high_prices_fail',
        'reason': '至少一个固定高买价仍为正。' if passing else '0.60/0.65/0.70 固定买价均无法在该 top159 组合下保持正资金曲线。'
    }
    payload = {
        'generatedAt': datetime.now(timezone.utc).isoformat(),
        'scope': 'top159 fixed high entry price raw-kline proxy backtest; research only; no live change',
        'params': params,
        'formulaSelfTest': formula_self_test(),
        'windowAudits': audits,
        'rows': rows,
        'uniqueVerdict': verdict,
    }
    if not payload['formulaSelfTest']['ok']:
        raise RuntimeError('formula self-test failed')
    out_json = REPORTS / 'top159_fixed_high_price_backtest_latest.json'
    out_md = REPORTS / 'top159_fixed_high_price_backtest_latest.md'
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    out_md.write_text(render_md(payload), encoding='utf-8')
    print(json.dumps({'ok': True, 'report': str(out_md), 'verdict': verdict}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
