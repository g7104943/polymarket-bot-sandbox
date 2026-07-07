#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import urllib.parse
import urllib.request
from urllib.parse import urlencode

ROOT = Path('/Users/mac/polyfun')
REPORTS = ROOT / 'reports'
GAMMA = 'https://gamma-api.polymarket.com'
CLOB = 'https://clob.polymarket.com'
OUT_DISCOVERY = REPORTS / 'event_market_discovery_truth_latest.json'
OUT_COMPARE = REPORTS / 'event_market_shadow_value_compare_latest.md'
OUT_COMPARE_JSON = REPORTS / 'event_market_shadow_value_compare_latest.json'
OUT_VERDICT = REPORTS / 'event_market_unique_verdict_latest.md'
OUT_VERDICT_JSON = REPORTS / 'event_market_unique_verdict_latest.json'

MAX_EVENTS = int(__import__('os').environ.get('EVENT_MARKET_MAX_EVENTS', '300'))
MAX_BOOKS = int(__import__('os').environ.get('EVENT_MARKET_MAX_BOOKS', '160'))
REQ_TIMEOUT = float(__import__('os').environ.get('EVENT_MARKET_TIMEOUT', '8'))

KEYWORDS = {
    'sports': ['nba','nfl','nhl','mlb','epl','soccer','football','basketball','baseball','tennis','ufc','golf','champions league','world cup','playoff','finals','win the game','which team'],
    'macro': ['fed','fomc','rate','interest','cpi','inflation','jobs report','unemployment','gdp','recession','tariff','treasury','ecb'],
    'weather': ['temperature','rain','snow','hurricane','tropical storm','weather','earthquake','wildfire','storm','tornado'],
    'onchain': ['bitcoin','ethereum','solana','xrp','btc','eth','crypto','block','hashrate','etf','stablecoin','usdt','usdc'],
}
SKIP_SUBJECTIVE = ['will trump say','tweet','mention','what will','who will be named','person of the year','approval rating']


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def bj_now() -> str:
    import zoneinfo
    return datetime.now(zoneinfo.ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S CST')


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    if params:
        url = url + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={'User-Agent': 'polyfun-event-shadow/1.0', 'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as resp:
        return json.loads(resp.read().decode('utf-8'))


def parse_jsonish(x: Any) -> Any:
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def classify(text: str, category: str = '') -> str | None:
    t = (text + ' ' + category).lower()
    if any(k in t for k in SKIP_SUBJECTIVE):
        return None
    for name, keys in KEYWORDS.items():
        if any(k in t for k in keys):
            return name
    return None


def best_book(token_id: str) -> dict[str, Any]:
    try:
        b = get_json(CLOB + '/book', {'token_id': token_id})
    except Exception as exc:
        return {'token_id': token_id, 'error': repr(exc)[:180]}
    bids = b.get('bids') or []
    asks = b.get('asks') or []
    def f(v):
        try: return float(v)
        except Exception: return math.nan
    # CLOB depth arrays are not guaranteed to be sorted best-first. In practice
    # some snapshots arrive bids low-to-high and asks high-to-low, so compute the
    # executable top of book explicitly.
    bid_prices = [f(x.get('price')) for x in bids if not math.isnan(f(x.get('price')))]
    ask_prices = [f(x.get('price')) for x in asks if not math.isnan(f(x.get('price')))]
    best_bid = max(bid_prices) if bid_prices else math.nan
    best_ask = min(ask_prices) if ask_prices else math.nan
    ask_depth_2c = 0.0
    ask_depth_5c = 0.0
    for a in asks:
        p = f(a.get('price')); s = f(a.get('size'))
        if math.isnan(p) or math.isnan(s) or math.isnan(best_ask):
            continue
        if p <= best_ask + 0.0200001:
            ask_depth_2c += s
        if p <= best_ask + 0.0500001:
            ask_depth_5c += s
    return {
        'token_id': token_id,
        'best_bid': best_bid,
        'best_ask': best_ask,
        'spread': round(best_ask - best_bid, 6) if not math.isnan(best_bid) and not math.isnan(best_ask) else None,
        'ask_depth_2c': round(ask_depth_2c, 4),
        'ask_depth_5c': round(ask_depth_5c, 4),
        'min_order_size': b.get('min_order_size'),
        'tick_size': b.get('tick_size'),
    }


def event_iter() -> list[dict[str, Any]]:
    out = []
    offset = 0
    limit = 100
    while len(out) < MAX_EVENTS:
        batch = get_json(GAMMA + '/events', {'active': 'true', 'closed': 'false', 'order': 'volume_24hr', 'ascending': 'false', 'limit': limit, 'offset': offset})
        if not isinstance(batch, list) or not batch:
            break
        out.extend(batch)
        offset += limit
        if len(batch) < limit:
            break
    return out[:MAX_EVENTS]


def scan() -> dict[str, Any]:
    events = event_iter()
    markets = []
    books_done = 0
    for ev in events:
        title = str(ev.get('title') or ev.get('ticker') or '')
        category = str(ev.get('category') or ev.get('subcategory') or '')
        ev_type = classify(title, category)
        if not ev_type:
            continue
        for m in ev.get('markets') or []:
            q = str(m.get('question') or m.get('title') or title)
            typ = classify(q + ' ' + title, category) or ev_type
            if typ not in {'sports','macro','weather','onchain'}:
                continue
            if not bool(m.get('enableOrderBook')):
                continue
            outcomes = parse_jsonish(m.get('outcomes') or [])
            token_ids = parse_jsonish(m.get('clobTokenIds') or m.get('clobTokenIds'.lower()) or [])
            prices = parse_jsonish(m.get('outcomePrices') or [])
            if not isinstance(outcomes, list) or not isinstance(token_ids, list) or len(token_ids) < 2:
                continue
            token_books = []
            for tok in token_ids[:2]:
                if books_done >= MAX_BOOKS:
                    break
                token_books.append(best_book(str(tok)))
                books_done += 1
                time.sleep(0.03)
            if len(token_books) < 2:
                continue
            usable_books = [b for b in token_books if b.get('best_ask') is not None and not math.isnan(float(b.get('best_ask', math.nan))) and b.get('spread') is not None]
            if not usable_books:
                continue
            best_side = min(usable_books, key=lambda b: float(b.get('best_ask', 9)))
            max_spread = max(float(b.get('spread') or 9) for b in usable_books)
            min_depth = min(float(b.get('ask_depth_2c') or 0) for b in usable_books)
            tradable_shadow = max_spread <= 0.05 and min_depth >= 5.0
            markets.append({
                'event_id': ev.get('id'), 'event_slug': ev.get('slug'), 'event_title': title,
                'market_id': m.get('id'), 'question': q, 'type': typ, 'endDate': m.get('endDate') or ev.get('endDate'),
                'volume': m.get('volume'), 'volume24hr': m.get('volume24hr') or m.get('volume_24hr'), 'liquidity': m.get('liquidity') or m.get('liquidityClob'),
                'outcomes': outcomes[:2], 'outcomePrices': prices[:2] if isinstance(prices, list) else prices,
                'books': token_books, 'best_shadow_side': best_side, 'max_spread': round(max_spread, 6), 'min_ask_depth_2c': round(min_depth, 4),
                'shadow_status': 'watch_candidate' if tradable_shadow else 'skip_liquidity_or_spread',
                'reason': 'needs_external_fair_probability_model; no live bet generated',
            })
            if books_done >= MAX_BOOKS:
                break
        if books_done >= MAX_BOOKS:
            break
    return {'generatedAt': now_iso(), 'beijingTime': bj_now(), 'eventsScanned': len(events), 'booksFetched': books_done, 'markets': markets}


def main() -> int:
    payload = scan()
    counts = {}
    for m in payload['markets']:
        counts[m['type']] = counts.get(m['type'], 0) + 1
    watch = [m for m in payload['markets'] if m['shadow_status'] == 'watch_candidate']
    summary = {
        'generatedAt': payload['generatedAt'], 'beijingTime': payload['beijingTime'], 'researchOnlyNoLiveChange': True,
        'sourceTruth': {
            'gamma': 'https://gamma-api.polymarket.com/events active=true closed=false order=volume_24hr',
            'clob': 'https://clob.polymarket.com/book public orderbook endpoint',
            'officialDocs': ['https://docs.polymarket.com/api-reference', 'https://docs.polymarket.com/trading/orderbook'],
        },
        'eventsScanned': payload['eventsScanned'], 'booksFetched': payload['booksFetched'], 'marketCountByType': counts,
        'watchCandidates': len(watch), 'approvedTrades': 0,
        'verdict': 'shadow_catalog_built_no_trade_signal; external fair probability models are required before betting',
    }
    write_json(OUT_DISCOVERY, payload)
    write_json(OUT_COMPARE_JSON, {'summary': summary, 'topWatch': watch[:50], 'allMarketsCount': len(payload['markets'])})
    write_json(OUT_VERDICT_JSON, summary)
    lines = [
        '# 事件型市场影子扫描', '', f"- 北京时间：`{summary['beijingTime']}`", '- 动作：`research_only_no_live_change`',
        f"- 扫描事件：`{summary['eventsScanned']}`，读取订单簿：`{summary['booksFetched']}`，影子观察候选：`{summary['watchCandidates']}`，真钱信号：`0`", '',
        '|类型|市场数|', '|---|---:|'
    ]
    for k,v in sorted(counts.items()):
        lines.append(f'|{k}|{v}|')
    lines += ['', '## 影子观察候选前20', '', '|类型|问题|最佳卖价|最大价差|2分钱深度|状态|', '|---|---|---:|---:|---:|---|']
    for m in watch[:20]:
        b=m.get('best_shadow_side') or {}
        lines.append(f"|{m.get('type')}|{str(m.get('question',''))[:80]}|{b.get('best_ask')}|{m.get('max_spread')}|{m.get('min_ask_depth_2c')}|{m.get('shadow_status')}|")
    lines += ['', '## 结论', '', '- 已建立事件型市场只读影子目录。', '- 当前没有真钱信号，因为还没有接入体育/宏观/天气的外部公允概率模型。', '- 下一步若继续事件路线，应先做其中一种事件类型的公允概率模型，而不是直接按市场价格下注。']
    txt='\n'.join(lines)+'\n'
    write_text(OUT_COMPARE, txt)
    write_text(OUT_VERDICT, txt)
    print(json.dumps({'compare': str(OUT_COMPARE), 'verdict': str(OUT_VERDICT), **summary}, ensure_ascii=False, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
