#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
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
ORDERBOOK = ROOT / 'data' / 'processed' / 'vnext_execution_orderbook_eth_usdt.jsonl'

spec = importlib.util.spec_from_file_location('crypto_search', BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules['crypto_search'] = base
spec.loader.exec_module(base)

spec2 = importlib.util.spec_from_file_location('fill_search', FILL_SCRIPT)
fill_search = importlib.util.module_from_spec(spec2)
assert spec2.loader is not None
sys.modules['fill_search'] = fill_search
spec2.loader.exec_module(fill_search)

ASSET = 'ETH'
TIMEFRAME = '15m'
WINDOWS = ['180d', '365d']
MAX_PRICES = [0.50,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58]
ORDER_TYPES = ['FOK','FAK','RESTING_REMAINDER']
START_BANKROLL = 850.0
STAKE_PCT = 0.01
ENTRY_START_SEC = 30
ENTRY_END_SEC = 180
MARKET_SEC = 15 * 60
MIN_DEPTH_MULT = 1.2


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def stable_hash(values: list[Any]) -> str:
    return hashlib.sha256('\n'.join(map(str, values)).encode()).hexdigest()[:16]


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


def candidate_frame(window: str, params: dict[str, Any]) -> pd.DataFrame:
    raw = base.load_raw(ASSET, TIMEFRAME)
    df, features = base.build_features(raw, TIMEFRAME)
    forbidden = [c for c in features if c in base.FORBIDDEN_FEATURES or any(c.startswith(p) for p in base.FORBIDDEN_PREFIXES)]
    if forbidden:
        raise RuntimeError(f'forbidden future features leaked: {forbidden}')
    train, val = fill_search.split_train_val(df, window, params['train_window'])
    feats = fill_search.feature_subset(features, params['feature_mode'])
    model = fill_search.fit_model(params['engine'], train, feats, params)
    if model is None:
        raise RuntimeError('top159 model failed to fit')
    prob = fill_search.predict(model, val, feats)
    dt, won, pred_up = fill_search.select_candidates(val, prob, params)
    out = pd.DataFrame({'dt': pd.to_datetime(dt, utc=True), 'won': won.astype(bool), 'pred_up': pred_up.astype(bool)})
    out['market_slug'] = out['dt'].map(lambda x: f"eth-updown-15m-{int(pd.Timestamp(x).timestamp())}")
    return out


def parse_book() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_slug: dict[str, list[dict[str, Any]]] = {}
    rows = 0
    depth_rows = 0
    token_ids: set[str] = set()
    if not ORDERBOOK.exists():
        return by_slug, {'path': str(ORDERBOOK), 'exists': False}
    with ORDERBOOK.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            slug = d.get('market_slug')
            ts = pd.to_datetime(d.get('ts') or d.get('generated_at'), utc=True, errors='coerce')
            if not slug or pd.isna(ts):
                continue
            rows += 1
            if d.get('token_id'):
                token_ids.add(str(d.get('token_id')))
            asks = d.get('asks_top') or []
            bids = d.get('bids_top') or []
            ask_depth = d.get('ask_depth_top3')
            bid_depth = d.get('bid_depth_top3')
            if ask_depth is None and asks:
                try:
                    ask_depth = sum(float(x.get('size', 0) or x.get('shares', 0) or 0) for x in asks[:3])
                except Exception:
                    ask_depth = None
            if bid_depth is None and bids:
                try:
                    bid_depth = sum(float(x.get('size', 0) or x.get('shares', 0) or 0) for x in bids[:3])
                except Exception:
                    bid_depth = None
            if ask_depth is not None:
                depth_rows += 1
            rec = {
                'ts': ts,
                'best_ask': _num(d.get('best_ask')),
                'best_bid': _num(d.get('best_bid')),
                'ask_depth': _num(ask_depth),
                'bid_depth': _num(bid_depth),
                'token_id': d.get('token_id'),
            }
            by_slug.setdefault(slug, []).append(rec)
    for rows_list in by_slug.values():
        rows_list.sort(key=lambda r: r['ts'])
    return by_slug, {
        'path': str(ORDERBOOK), 'exists': True, 'rows': rows, 'markets': len(by_slug),
        'tokenIds': len(token_ids), 'rowsWithAskDepth': depth_rows,
        'strictDirectionMapping': False,
        'directionMappingRisk': 'local orderbook has market_slug/token_id but no explicit UP/DOWN outcome label; strict matching to predicted side is unavailable',
    }


def _num(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        return None
    return None


def market_start(slug: str) -> pd.Timestamp | None:
    try:
        sec = int(str(slug).rsplit('-', 1)[-1])
    except Exception:
        return None
    return pd.to_datetime(sec, unit='s', utc=True)


def simulate(cands: pd.DataFrame, book: dict[str, list[dict[str, Any]]], order_type: str, max_price: float) -> dict[str, Any]:
    eq = START_BANKROLL
    peak = eq
    maxdd = 0.0
    wins = losses = filled = failures = 0
    winner_candidates = int(cands['won'].sum())
    loser_candidates = int((~cands['won']).sum())
    winner_filled = loser_filled = 0
    fill_fracs = []
    prices = []
    pnl_values = []
    covered_markets = 0
    strict_entry_covered = 0
    any_book_covered = 0
    for row in cands.itertuples(index=False):
        slug = row.market_slug
        rows = book.get(slug, [])
        if rows:
            any_book_covered += 1
        start = market_start(slug)
        if start is None:
            failures += 1
            continue
        entry_rows = [r for r in rows if start + pd.Timedelta(seconds=ENTRY_START_SEC) <= r['ts'] <= start + pd.Timedelta(seconds=ENTRY_END_SEC)]
        if entry_rows:
            strict_entry_covered += 1
        all_valid_rows = [r for r in rows if start + pd.Timedelta(seconds=ENTRY_START_SEC) <= r['ts'] <= start + pd.Timedelta(seconds=MARKET_SEC)]
        chosen_rows = entry_rows if order_type in {'FOK','FAK'} else all_valid_rows
        if not chosen_rows:
            failures += 1
            fill_fracs.append(0.0)
            continue
        covered_markets += 1
        notional = eq * STAKE_PCT
        fill_fraction, avg_price = fill_order(chosen_rows, order_type, max_price, notional)
        if fill_fraction <= 0 or avg_price is None:
            failures += 1
            fill_fracs.append(0.0)
            continue
        ret = (1.0 / avg_price - 1.0) if bool(row.won) else -1.0
        stake = eq * STAKE_PCT * fill_fraction
        eq += stake * ret
        eq = max(0.0, eq)
        peak = max(peak, eq)
        maxdd = max(maxdd, peak - eq)
        filled += 1
        prices.append(avg_price)
        fill_fracs.append(fill_fraction)
        pnl_values.append(stake * ret)
        if bool(row.won):
            wins += 1; winner_filled += 1
        else:
            losses += 1; loser_filled += 1
    return {
        'orderType': order_type,
        'maxEntryPrice': max_price,
        'requestedTrades': int(len(cands)),
        'orderbookAnyCoverageCount': any_book_covered,
        'strictEntryCoverageCount': strict_entry_covered,
        'coverageCountUsed': covered_markets,
        'orderbookCoveragePct': round(100.0 * any_book_covered / len(cands), 6) if len(cands) else 0.0,
        'strictEntryCoveragePct': round(100.0 * strict_entry_covered / len(cands), 6) if len(cands) else 0.0,
        'filledTrades': filled,
        'wins': wins,
        'losses': losses,
        'winRatePct': round(100.0 * wins / filled, 6) if filled else 0.0,
        'endingFunds': round(eq, 6),
        'compoundPnl': round(eq - START_BANKROLL, 6),
        'maxDrawdownUsd': round(maxdd, 6),
        'maxDrawdownPct': round(100.0 * maxdd / START_BANKROLL, 6),
        'winnerFillRatePct': round(100.0 * winner_filled / winner_candidates, 6) if winner_candidates else 0.0,
        'loserFillRatePct': round(100.0 * loser_filled / loser_candidates, 6) if loser_candidates else 0.0,
        'avgBuyPrice': round(float(np.mean(prices)), 6) if prices else 0.0,
        'failureCount': failures,
        'setHash': stable_hash([order_type, max_price, len(cands), filled, wins, losses, round(eq, 6)]),
        'pnlSampleCount': len(pnl_values),
    }


def fill_order(rows: list[dict[str, Any]], order_type: str, max_price: float, notional: float) -> tuple[float, float | None]:
    if order_type == 'FOK':
        r = first_eligible(rows, max_price)
        if not r:
            return 0.0, None
        ask = r['best_ask']; depth = r['ask_depth']
        if ask is None or depth is None or ask <= 0:
            return 0.0, None
        need_shares = notional / ask
        if depth >= need_shares * MIN_DEPTH_MULT:
            return 1.0, ask
        return 0.0, None
    if order_type == 'FAK':
        r = first_eligible(rows, max_price)
        if not r:
            return 0.0, None
        ask = r['best_ask']; depth = r['ask_depth']
        if ask is None or depth is None or ask <= 0:
            return 0.0, None
        need_shares = notional / ask
        return min(1.0, max(0.0, depth / need_shares)), ask
    # RESTING_REMAINDER: consume depth across subsequent local snapshots, capped at full notional.
    remaining = notional
    cost = 0.0
    filled_notional = 0.0
    for r in rows:
        ask = r['best_ask']; depth = r['ask_depth']
        if ask is None or depth is None or ask <= 0 or ask > max_price:
            continue
        available_notional = depth * ask
        take = min(remaining, available_notional)
        if take <= 0:
            continue
        cost += take
        filled_notional += take
        remaining -= take
        if remaining <= 1e-9:
            break
    if filled_notional <= 0:
        return 0.0, None
    return min(1.0, filled_notional / notional), cost / (filled_notional / (cost / filled_notional) if False else filled_notional) if False else weighted_price_placeholder(rows, max_price, notional)


def weighted_price_placeholder(rows: list[dict[str, Any]], max_price: float, notional: float) -> float | None:
    remaining = notional
    shares = 0.0
    cost = 0.0
    for r in rows:
        ask = r['best_ask']; depth = r['ask_depth']
        if ask is None or depth is None or ask <= 0 or ask > max_price:
            continue
        take_shares = min(depth, remaining / ask)
        if take_shares <= 0:
            continue
        shares += take_shares
        cost += take_shares * ask
        remaining -= take_shares * ask
        if remaining <= 1e-9:
            break
    return cost / shares if shares > 0 else None


def first_eligible(rows: list[dict[str, Any]], max_price: float) -> dict[str, Any] | None:
    for r in rows:
        ask = r['best_ask']
        if ask is not None and ask <= max_price:
            return r
    return None


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    params = top159_params()
    book, book_audit = parse_book()
    rows = []
    cands_by_window = {w: candidate_frame(w, params) for w in WINDOWS}
    for w, cands in cands_by_window.items():
        for typ in ORDER_TYPES:
            for price in MAX_PRICES:
                row = simulate(cands, book, typ, price)
                row['window'] = w
                rows.append(row)
    # Selection uses only strict local orderbook data. If coverage is too small, no live candidate is claimed.
    eligible = [r for r in rows if r['endingFunds'] > START_BANKROLL and ((r['window']=='180d' and r['requestedTrades']>=100) or (r['window']=='365d' and r['requestedTrades']>=200))]
    by_key: dict[tuple[str,float], dict[str, Any]] = {}
    for typ in ORDER_TYPES:
        for price in MAX_PRICES:
            pair = [r for r in rows if r['orderType']==typ and abs(r['maxEntryPrice']-price)<1e-9]
            if len(pair)==2:
                by_key[(typ,price)] = {'180d': next(r for r in pair if r['window']=='180d'), '365d': next(r for r in pair if r['window']=='365d')}
    passing = []
    for key, pair in by_key.items():
        if pair['180d']['endingFunds'] > START_BANKROLL and pair['365d']['endingFunds'] > START_BANKROLL and pair['180d']['filledTrades'] >= 100 and pair['365d']['filledTrades'] >= 200 and pair['180d']['winnerFillRatePct'] + 1e-9 >= pair['180d']['loserFillRatePct'] - 2.0 and pair['365d']['winnerFillRatePct'] + 1e-9 >= pair['365d']['loserFillRatePct'] - 2.0:
            passing.append((key,pair))
    selected = None
    if passing:
        selected = max(passing, key=lambda x: (x[1]['180d']['endingFunds'] + x[1]['365d']['endingFunds'] - x[1]['365d']['maxDrawdownUsd']))
    verdict = {
        'status': 'candidate_found' if selected else 'no_live_candidate_from_local_orderbook',
        'selected': {'orderType': selected[0][0], 'maxEntryPrice': selected[0][1]} if selected else None,
        'reason': '本地订单簿严格入场窗口覆盖不足或候选不达标；不能用这批本地订单簿给出真钱订单类型/最高买价定论。' if not selected else '本地订单簿严格回放通过。',
    }
    payload = {
        'generatedAt': now_iso(),
        'scope': 'top159 local historical orderbook hyperopt; research only; no live',
        'params': params,
        'bookAudit': book_audit,
        'candidateCounts': {w: len(cands) for w,cands in cands_by_window.items()},
        'maxPrices': MAX_PRICES,
        'orderTypes': ORDER_TYPES,
        'rows': rows,
        'uniqueVerdict': verdict,
    }
    write(REPORTS / 'top159_local_orderbook_hyperopt_latest.json', payload)
    (REPORTS / 'top159_local_orderbook_hyperopt_latest.md').write_text(render_md(payload), encoding='utf-8')
    print(json.dumps({'ok': True, 'verdict': verdict, 'report': str(REPORTS / 'top159_local_orderbook_hyperopt_latest.md')}, ensure_ascii=False, indent=2))


def render_md(payload: dict[str, Any]) -> str:
    lines = ['# top159 本地历史订单簿超参', '', f"生成时间：`{payload['generatedAt']}`", '']
    audit = payload['bookAudit']
    lines.append('## 数据覆盖')
    lines.append(f"- 订单簿文件：`{audit.get('path')}`")
    lines.append(f"- 行数：`{audit.get('rows',0)}`，市场数：`{audit.get('markets',0)}`，有深度行：`{audit.get('rowsWithAskDepth',0)}`")
    lines.append(f"- 方向映射：`{audit.get('strictDirectionMapping')}`，说明：{audit.get('directionMappingRisk')}")
    lines.append('')
    lines.append('## 结果表')
    lines.append('|窗口|订单类型|最高买价|交易数|成交数|胜/负|胜率|期末资金|复利盈亏|最大回撤|赢家成交率|输家成交率|平均买价|失败次数|订单簿覆盖率|严格入场覆盖率|')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    # Show all threshold rows, ordered by window/type/price.
    for r in sorted(payload['rows'], key=lambda x: (x['window'], x['orderType'], x['maxEntryPrice'])):
        lines.append(f"|{r['window']}|{r['orderType']}|{r['maxEntryPrice']:.2f}|{r['requestedTrades']}|{r['filledTrades']}|{r['wins']}/{r['losses']}|{r['winRatePct']:.2f}%|{r['endingFunds']:.2f}|{r['compoundPnl']:.2f}|{r['maxDrawdownUsd']:.2f}|{r['winnerFillRatePct']:.2f}%|{r['loserFillRatePct']:.2f}%|{r['avgBuyPrice']:.4f}|{r['failureCount']}|{r['orderbookCoveragePct']:.4f}%|{r['strictEntryCoveragePct']:.4f}%|")
    lines.append('')
    lines.append('## 唯一结论')
    v = payload['uniqueVerdict']
    lines.append(f"- 状态：`{v['status']}`")
    lines.append(f"- 选择：`{v.get('selected')}`")
    lines.append(f"- 原因：{v['reason']}")
    lines.append('- 这份本地订单簿目前只能证明覆盖情况和模拟边界；若严格入场窗口没有足够覆盖，不能拿它给真钱定最高买价。')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    main()
