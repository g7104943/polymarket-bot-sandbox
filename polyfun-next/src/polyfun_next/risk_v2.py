from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

BEIJING_TZ = timezone(timedelta(hours=8))
ROOT = Path('/Users/mac/polyfun')
NEXT = ROOT / 'polyfun-next'
RUNTIME = NEXT / 'runtime'
REPORTS = ROOT / 'reports'
LEDGER = RUNTIME / 'canary_ledger.jsonl'
SETTLEMENT = REPORTS / 'top159_monitor_official_settlement_fix_latest.json'
DEFAULT_PROFILE = RUNTIME / 'top159_risk_profile_v2.json'
SHADOW_REPORT = REPORTS / 'top159_risk_v2_shadow_latest.json'
V3_PROFILE = RUNTIME / 'top159_risk_profile_v3.json'
V3_SHADOW_REPORT = REPORTS / 'top159_risk_v3_shadow_latest.json'
V4_PROFILE = RUNTIME / 'top159_risk_profile_v4.json'
V4_SHADOW_REPORT = REPORTS / 'top159_risk_v4_shadow_latest.json'


@dataclass(frozen=True)
class OfficialTrade:
    market_slug: str
    dt: datetime
    first_ts: str | None
    order_ids: tuple[str, ...]
    side: str
    won: bool
    actual_cost: float
    actual_shares: float
    actual_avg_price: float
    actual_pnl: float
    model_score: float | None = None
    shock_candidate_id: str | None = None
    strategy_profile: str | None = None


@dataclass(frozen=True)
class RiskV2Policy:
    name: str
    family: str = 'combo'
    update_lag_bars: int = 2
    loss_streak_n: int = 0
    loss_pause_bars: int = 0
    rolling_window: int = 0
    rolling_min_winrate: float | None = None
    rolling_min_pnl: float | None = None
    rolling_pause_bars: int = 0
    day_drawdown_fraction: float | None = None
    day_pause_bars: int = 0
    global_drawdown_fraction: float | None = None
    global_pause_bars: int = 0
    high_price_min: float | None = None
    high_price_loss_streak_n: int = 0
    high_price_pause_bars: int = 0
    low_score_max: float | None = None
    low_score_loss_streak_n: int = 0
    low_score_pause_bars: int = 0
    min_retention_rate: float = 0.50
    state_yellow_window: int = 0
    state_yellow_min_pnl: float | None = None
    state_yellow_pause_bars: int = 0
    state_red_window: int = 0
    state_red_min_pnl: float | None = None
    state_red_pause_bars: int = 0
    state_red_loss_streak_n: int = 0
    state_red_high_price_min: float | None = None
    state_red_high_price_loss_streak_n: int = 0
    state_red_score_price_min: float | None = None
    state_red_score_price_max: float | None = None
    state_red_score_price_loss_streak_n: int = 0
    state_red_day_drawdown_fraction: float | None = None
    state_red_global_drawdown_fraction: float | None = None


@dataclass(frozen=True)
class RiskV2Profile:
    enabled: bool = True
    shadow_only: bool = True
    profile: str = 'top159_real_order_risk_v2_shadow'
    candidate_id: str = ''
    strict_061_only: bool = True
    selected_policy: RiskV2Policy | None = None


@dataclass(frozen=True)
class RiskV2Decision:
    enabled: bool
    shadow_only: bool
    allowed: bool
    action: str
    reason: str
    policy: dict[str, Any] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


def parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def market_start_from_slug(slug: str) -> datetime | None:
    try:
        ts = int(str(slug).rsplit('-', 1)[1])
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        return None


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return default


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    tmp.replace(path)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def load_order_candidate_map(ledger_path: Path = LEDGER) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for obj in iter_jsonl(ledger_path):
        if obj.get('event_type') not in {'order_official_confirmation', 'position_filled_holding_to_settlement', 'order_post_result'}:
            continue
        payload = obj.get('payload') or {}
        if not isinstance(payload, dict):
            continue
        status = payload.get('status') if isinstance(payload.get('status'), dict) else {}
        raw = status.get('raw') if isinstance(status.get('raw'), dict) else {}
        order_id = status.get('order_id') or raw.get('id') or raw.get('orderID')
        if not order_id:
            continue
        cand = payload.get('candidate') if isinstance(payload.get('candidate'), dict) else None
        plan = payload.get('plan') if isinstance(payload.get('plan'), dict) else {}
        if cand is None and isinstance(plan.get('candidate'), dict):
            cand = plan.get('candidate')
        if not isinstance(cand, dict) or not cand:
            continue
        prev = out.get(str(order_id))
        if prev is None or obj.get('event_type') != 'order_post_result':
            out[str(order_id)] = cand
    return out


def load_official_trades(
    *,
    settlement_path: Path = SETTLEMENT,
    ledger_path: Path = LEDGER,
    strict_061_only: bool = True,
) -> list[OfficialTrade]:
    settlement = read_json(settlement_path, {}) or {}
    order_to_candidate = load_order_candidate_map(ledger_path)
    rows: list[OfficialTrade] = []
    for detail in settlement.get('details') or []:
        if not isinstance(detail, dict):
            continue
        order_ids = tuple(str(x) for x in (detail.get('order_ids') or []))
        candidates = [order_to_candidate[o] for o in order_ids if o in order_to_candidate]
        if strict_061_only:
            candidates = [c for c in candidates if c.get('shock_candidate_id') == '06173b68b0d86431']
            if not candidates:
                continue
        elif not candidates:
            continue
        cand = candidates[-1]
        settlement_obj = detail.get('settlement') if isinstance(detail.get('settlement'), dict) else {}
        result = str(settlement_obj.get('result') or '').lower()
        if result not in {'win', 'loss'}:
            continue
        slug = str(detail.get('market_slug') or '')
        dt = market_start_from_slug(slug)
        if dt is None:
            continue
        prices = [float(x) for x in (detail.get('prices') or [])]
        shares = [float(x) for x in (detail.get('shares') or [])]
        if not prices or len(prices) != len(shares) or sum(shares) <= 0:
            continue
        cost = sum(p * s for p, s in zip(prices, shares))
        total_shares = sum(shares)
        avg_price = cost / total_shares
        actual_pnl = total_shares - cost if result == 'win' else -cost
        try:
            model_score = float(cand.get('model_score'))
        except Exception:
            model_score = None
        side = str(cand.get('side') or detail.get('outcome') or '').upper()
        rows.append(OfficialTrade(
            market_slug=slug,
            dt=dt,
            first_ts=detail.get('first_ts'),
            order_ids=order_ids,
            side=side,
            won=result == 'win',
            actual_cost=float(cost),
            actual_shares=float(total_shares),
            actual_avg_price=float(avg_price),
            actual_pnl=float(actual_pnl),
            model_score=model_score,
            shock_candidate_id=cand.get('shock_candidate_id'),
            strategy_profile=cand.get('strategy_profile'),
        ))
    # One logical trade per market, keep the latest detail if duplicates exist.
    by_slug: dict[str, OfficialTrade] = {}
    for row in sorted(rows, key=lambda r: (r.dt, r.first_ts or '')):
        by_slug[row.market_slug] = row
    return sorted(by_slug.values(), key=lambda r: r.dt)


def load_risk_v2_profile(path: Path = DEFAULT_PROFILE) -> RiskV2Profile:
    raw = read_json(path, {}) or {}
    if not raw:
        return RiskV2Profile(enabled=False, shadow_only=True)
    policy_raw = raw.get('selected_policy') or raw.get('selectedPolicy') or raw.get('policy')
    if isinstance(policy_raw, dict):
        allowed = set(RiskV2Policy.__dataclass_fields__)
        policy = RiskV2Policy(**{k: v for k, v in policy_raw.items() if k in allowed})
    else:
        policy = None
    return RiskV2Profile(
        enabled=bool(raw.get('enabled', True)),
        shadow_only=bool(raw.get('shadow_only', raw.get('shadowOnly', True))),
        profile=str(raw.get('profile') or 'top159_real_order_risk_v2_shadow'),
        candidate_id=str(raw.get('candidate_id') or raw.get('candidateId') or ''),
        strict_061_only=bool(raw.get('strict_061_only', True)),
        selected_policy=policy,
    )


def _bj_day(dt: datetime) -> str:
    return dt.astimezone(BEIJING_TZ).date().isoformat()


def _release_pending(pending: list[tuple[datetime, OfficialTrade]], now_dt: datetime, state: dict[str, Any]) -> None:
    ready = [x for x in pending if x[0] <= now_dt]
    pending[:] = [x for x in pending if x[0] > now_dt]
    for release_at, trade in sorted(ready, key=lambda x: x[0]):
        pnl = trade.actual_pnl
        state['funds'] += pnl
        state['global_high'] = max(state['global_high'], state['funds'])
        day = _bj_day(trade.dt)
        if state.get('day_key') != day:
            state['day_key'] = day
            state['day_high'] = state['funds']
        state['day_high'] = max(state.get('day_high', state['funds']), state['funds'])
        state['settled_count'] += 1
        state['last_settlement_time'] = release_at
        state['official_pnl'] += pnl
        state['curve'].append(state['funds'])
        state['max_drawdown'] = max(state['max_drawdown'], state['global_high'] - state['funds'])
        state['rolling_results'].append(1 if trade.won else 0)
        state['rolling_pnls'].append(pnl)
        state['rolling_prices'].append(trade.actual_avg_price)
        state['rolling_scores'].append(trade.model_score or 0.0)
        if len(state['rolling_results']) > 200:
            state['rolling_results'] = state['rolling_results'][-200:]
            state['rolling_pnls'] = state['rolling_pnls'][-200:]
            state['rolling_prices'] = state['rolling_prices'][-200:]
            state['rolling_scores'] = state['rolling_scores'][-200:]
        if trade.won:
            state['loss_streak'] = 0
            state['high_price_loss_streak'] = 0
            state['low_score_loss_streak'] = 0
        else:
            state['loss_streak'] += 1
            if state.get('high_price_min') is not None and trade.actual_avg_price >= state['high_price_min']:
                state['high_price_loss_streak'] += 1
            else:
                state['high_price_loss_streak'] = 0
            if state.get('low_score_max') is not None and trade.model_score is not None and trade.model_score <= state['low_score_max']:
                state['low_score_loss_streak'] += 1
            else:
                state['low_score_loss_streak'] = 0
            score_price_hit = (
                state.get('score_price_min') is not None
                and state.get('score_price_max') is not None
                and trade.model_score is not None
                and trade.actual_avg_price >= state['score_price_min']
                and trade.model_score <= state['score_price_max']
            )
            if score_price_hit:
                state['score_price_loss_streak'] += 1
            else:
                state['score_price_loss_streak'] = 0


def _active_pause_until(state: dict[str, Any], now_dt: datetime) -> bool:
    pause_until = state.get('pause_until')
    return isinstance(pause_until, datetime) and pause_until > now_dt


def _maybe_trigger(policy: RiskV2Policy, state: dict[str, Any], now_dt: datetime) -> tuple[bool, str, int]:
    if _active_pause_until(state, now_dt):
        return True, str(state.get('pause_reason') or 'pause_active'), int(state.get('pause_bars') or 0)
    triggers: list[tuple[str, int]] = []
    if policy.family == 'state_machine':
        if policy.state_yellow_window and len(state['rolling_pnls']) >= policy.state_yellow_window and policy.state_yellow_min_pnl is not None:
            psum = sum(state['rolling_pnls'][-policy.state_yellow_window:])
            if psum <= policy.state_yellow_min_pnl:
                triggers.append((f'yellow_rolling_pnl_{psum:.2f}<={policy.state_yellow_min_pnl}', policy.state_yellow_pause_bars))
        if policy.state_red_window and len(state['rolling_pnls']) >= policy.state_red_window and policy.state_red_min_pnl is not None:
            psum = sum(state['rolling_pnls'][-policy.state_red_window:])
            if psum <= policy.state_red_min_pnl:
                triggers.append((f'red_rolling_pnl_{psum:.2f}<={policy.state_red_min_pnl}', policy.state_red_pause_bars))
        if policy.state_red_loss_streak_n and state['loss_streak'] >= policy.state_red_loss_streak_n:
            triggers.append((f'red_loss_streak_{state["loss_streak"]}>={policy.state_red_loss_streak_n}', policy.state_red_pause_bars))
        if policy.state_red_high_price_loss_streak_n and state['high_price_loss_streak'] >= policy.state_red_high_price_loss_streak_n:
            triggers.append((f'red_high_price_loss_streak_{state["high_price_loss_streak"]}>={policy.state_red_high_price_loss_streak_n}', policy.state_red_pause_bars))
        if policy.state_red_score_price_loss_streak_n and state.get('score_price_loss_streak', 0) >= policy.state_red_score_price_loss_streak_n:
            triggers.append((f'red_score_price_loss_streak_{state["score_price_loss_streak"]}>={policy.state_red_score_price_loss_streak_n}', policy.state_red_pause_bars))
        day_high = float(state.get('day_high') or state['funds'])
        if policy.state_red_day_drawdown_fraction is not None and day_high > 0:
            dd = (day_high - state['funds']) / day_high
            if dd >= policy.state_red_day_drawdown_fraction:
                triggers.append((f'red_day_drawdown_{dd:.3f}>={policy.state_red_day_drawdown_fraction}', policy.state_red_pause_bars))
        global_high = float(state.get('global_high') or state['funds'])
        if policy.state_red_global_drawdown_fraction is not None and global_high > 0:
            dd = (global_high - state['funds']) / global_high
            if dd >= policy.state_red_global_drawdown_fraction:
                triggers.append((f'red_global_drawdown_{dd:.3f}>={policy.state_red_global_drawdown_fraction}', policy.state_red_pause_bars))
    if policy.loss_streak_n and state['loss_streak'] >= policy.loss_streak_n:
        triggers.append((f'loss_streak_{state["loss_streak"]}>={policy.loss_streak_n}', policy.loss_pause_bars))
    if policy.rolling_window and len(state['rolling_results']) >= policy.rolling_window:
        results = state['rolling_results'][-policy.rolling_window:]
        pnls = state['rolling_pnls'][-policy.rolling_window:]
        wr = sum(results) / len(results) if results else 0.0
        psum = sum(pnls)
        if policy.rolling_min_winrate is not None and wr <= policy.rolling_min_winrate:
            triggers.append((f'rolling_winrate_{wr:.3f}<={policy.rolling_min_winrate}', policy.rolling_pause_bars))
        if policy.rolling_min_pnl is not None and psum <= policy.rolling_min_pnl:
            triggers.append((f'rolling_pnl_{psum:.2f}<={policy.rolling_min_pnl}', policy.rolling_pause_bars))
    day_high = float(state.get('day_high') or state['funds'])
    if policy.day_drawdown_fraction is not None and day_high > 0:
        dd = (day_high - state['funds']) / day_high
        if dd >= policy.day_drawdown_fraction:
            triggers.append((f'day_drawdown_{dd:.3f}>={policy.day_drawdown_fraction}', policy.day_pause_bars))
    global_high = float(state.get('global_high') or state['funds'])
    if policy.global_drawdown_fraction is not None and global_high > 0:
        dd = (global_high - state['funds']) / global_high
        if dd >= policy.global_drawdown_fraction:
            triggers.append((f'global_drawdown_{dd:.3f}>={policy.global_drawdown_fraction}', policy.global_pause_bars))
    if policy.high_price_min is not None and policy.high_price_loss_streak_n and state['high_price_loss_streak'] >= policy.high_price_loss_streak_n:
        triggers.append((f'high_price_loss_streak_{state["high_price_loss_streak"]}>={policy.high_price_loss_streak_n}', policy.high_price_pause_bars))
    if policy.low_score_max is not None and policy.low_score_loss_streak_n and state['low_score_loss_streak'] >= policy.low_score_loss_streak_n:
        triggers.append((f'low_score_loss_streak_{state["low_score_loss_streak"]}>={policy.low_score_loss_streak_n}', policy.low_score_pause_bars))
    if not triggers:
        return False, 'pass', 0
    reason, bars = max(triggers, key=lambda x: x[1])
    bars = max(1, int(bars or 1))
    pause_start = state.get('last_settlement_time')
    if not isinstance(pause_start, datetime):
        pause_start = now_dt
    pause_until = pause_start + timedelta(minutes=15 * bars)
    if pause_until <= now_dt:
        return False, 'pass', 0
    state['pause_until'] = pause_until
    state['pause_reason'] = reason
    state['pause_bars'] = bars
    state['pause_count'] += 1
    state['pause_total_bars'] += bars
    return True, reason, bars


def simulate_official_risk_policy(
    trades: list[OfficialTrade],
    policy: RiskV2Policy | None,
    *,
    initial_funds: float = 847.091209,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        'funds': float(initial_funds),
        'global_high': float(initial_funds),
        'day_key': None,
        'day_high': float(initial_funds),
        'loss_streak': 0,
        'high_price_loss_streak': 0,
        'low_score_loss_streak': 0,
        'score_price_loss_streak': 0,
        'rolling_results': [],
        'rolling_pnls': [],
        'rolling_prices': [],
        'rolling_scores': [],
        'curve': [],
        'max_drawdown': 0.0,
        'official_pnl': 0.0,
        'settled_count': 0,
        'pause_until': None,
        'pause_reason': None,
        'pause_bars': 0,
        'pause_count': 0,
        'pause_total_bars': 0,
        'last_settlement_time': None,
        'high_price_min': (policy.state_red_high_price_min if policy and policy.state_red_high_price_min is not None else (policy.high_price_min if policy else None)),
        'score_price_min': policy.state_red_score_price_min if policy else None,
        'score_price_max': policy.state_red_score_price_max if policy else None,
        'low_score_max': policy.low_score_max if policy else None,
    }
    pending: list[tuple[datetime, OfficialTrade]] = []
    kept: list[OfficialTrade] = []
    skipped: list[tuple[OfficialTrade, str]] = []
    max_loss_streak_seen = 0
    for trade in sorted(trades, key=lambda r: r.dt):
        if state.get('day_key') != _bj_day(trade.dt):
            state['day_key'] = _bj_day(trade.dt)
            state['day_high'] = state['funds']
        _release_pending(pending, trade.dt, state)
        max_loss_streak_seen = max(max_loss_streak_seen, int(state['loss_streak']))
        if policy is not None:
            blocked, reason, _bars = _maybe_trigger(policy, state, trade.dt)
        else:
            blocked, reason = False, 'pass'
        if blocked:
            skipped.append((trade, reason))
            continue
        kept.append(trade)
        lag = max(0, int(policy.update_lag_bars if policy else 0))
        release_at = trade.dt + timedelta(minutes=15 * lag)
        pending.append((release_at, trade))
    future = datetime.max.replace(tzinfo=timezone.utc)
    _release_pending(pending, future, state)
    max_loss_streak_seen = max(max_loss_streak_seen, int(state['loss_streak']))
    wins = sum(1 for t in kept if t.won)
    losses = len(kept) - wins
    skipped_w = sum(1 for t, _ in skipped if t.won)
    skipped_l = len(skipped) - skipped_w
    all_cost = sum(t.actual_cost for t in kept)
    all_shares = sum(t.actual_shares for t in kept)
    win_prices = [t.actual_avg_price for t in kept if t.won]
    loss_prices = [t.actual_avg_price for t in kept if not t.won]
    reasons: dict[str, int] = {}
    for _t, r in skipped:
        reasons[r] = reasons.get(r, 0) + 1
    return {
        'policyName': policy.name if policy else 'baseline_061_official_actual',
        'allTrades': len(trades),
        'keptTrades': len(kept),
        'skippedTrades': len(skipped),
        'wins': wins,
        'losses': losses,
        'winRatePct': round(100.0 * wins / len(kept), 6) if kept else 0.0,
        'officialActualPnl': round(float(state['official_pnl']), 6),
        'endingFunds': round(float(state['funds']), 6),
        'maxDrawdownUsd': round(float(state['max_drawdown']), 6),
        'returnDrawdownRatio': round(float(state['official_pnl']) / state['max_drawdown'], 6) if state['max_drawdown'] > 1e-12 else (999.0 if state['official_pnl'] > 0 else 0.0),
        'retentionRatePct': round(100.0 * len(kept) / max(1, len(trades)), 6),
        'interceptedWinners': skipped_w,
        'interceptedLosers': skipped_l,
        'pauseCount': int(state['pause_count']),
        'pauseTotalBars': int(state['pause_total_bars']),
        'longestSettledLossStreak': int(max_loss_streak_seen),
        'avgBuyPrice': round(sum(t.actual_avg_price for t in kept) / len(kept), 6) if kept else 0.0,
        'weightedAvgBuyPrice': round(all_cost / all_shares, 6) if all_shares > 0 else 0.0,
        'winnerAvgBuyPrice': round(sum(win_prices) / len(win_prices), 6) if win_prices else 0.0,
        'loserAvgBuyPrice': round(sum(loss_prices) / len(loss_prices), 6) if loss_prices else 0.0,
        'skipReasons': reasons,
        'setHash': stable_hash([(t.market_slug, t.won, round(t.actual_avg_price, 6), r) for t, r in skipped] + [(t.market_slug, t.won, round(t.actual_avg_price, 6), 'kept') for t in kept]),
    }


def stable_hash(parts: Any) -> str:
    import hashlib
    h = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False).encode('utf-8')).hexdigest()
    return h[:16]


def evaluate_live_shadow_decision(
    candidate: dict[str, Any] | None,
    *,
    profile_path: Path = DEFAULT_PROFILE,
    settlement_path: Path = SETTLEMENT,
    ledger_path: Path = LEDGER,
) -> RiskV2Decision:
    profile = load_risk_v2_profile(profile_path)
    if not profile.enabled or profile.selected_policy is None:
        return RiskV2Decision(enabled=False, shadow_only=True, allowed=True, action='disabled', reason='risk_v2_disabled')
    candidate_start = None
    if isinstance(candidate, dict):
        candidate_start = parse_utc(str(candidate.get('candidate_start') or candidate.get('market_start') or candidate.get('generated_at') or ''))
    if candidate_start is None:
        candidate_start = datetime.now(timezone.utc)
    rows = [t for t in load_official_trades(settlement_path=settlement_path, ledger_path=ledger_path, strict_061_only=profile.strict_061_only) if t.dt < candidate_start]
    # Replay the selected policy up to the current candidate. If a policy would
    # already be paused now, the next candidate is shadow-blocked.
    state_policy = profile.selected_policy
    result = simulate_official_risk_policy(rows, state_policy, initial_funds=847.091209)
    # Re-evaluate the final state by appending a harmless synthetic trade time is
    # intentionally avoided; instead infer active pause from last skipped reason.
    # For live shadow, a recent policy skip at or after the latest trade implies
    # a current block only if the pause window still extends over candidate_start.
    # This uses a direct chronological replay with an empty current trade below.
    paused, reason = _current_pause_after_replay(rows, state_policy, candidate_start)
    action = 'shadow_block' if paused else 'shadow_pass'
    return RiskV2Decision(
        enabled=True,
        shadow_only=profile.shadow_only,
        allowed=not paused,
        action=action if profile.shadow_only else ('block' if paused else 'pass'),
        reason=reason if paused else 'pass',
        policy=asdict(state_policy),
        metrics={
            'profile': profile.profile,
            'candidate_id': profile.candidate_id,
            'strict_061_only': profile.strict_061_only,
            'past_official_trades': len(rows),
            'shadow_replay': result,
            'candidate_start': candidate_start.isoformat(),
        },
    )


def _current_pause_after_replay(rows: list[OfficialTrade], policy: RiskV2Policy, now_dt: datetime) -> tuple[bool, str]:
    state: dict[str, Any] = {
        'funds': 847.091209,
        'global_high': 847.091209,
        'day_key': None,
        'day_high': 847.091209,
        'loss_streak': 0,
        'high_price_loss_streak': 0,
        'low_score_loss_streak': 0,
        'score_price_loss_streak': 0,
        'rolling_results': [],
        'rolling_pnls': [],
        'rolling_prices': [],
        'rolling_scores': [],
        'curve': [],
        'max_drawdown': 0.0,
        'official_pnl': 0.0,
        'settled_count': 0,
        'pause_until': None,
        'pause_reason': None,
        'pause_bars': 0,
        'pause_count': 0,
        'pause_total_bars': 0,
        'last_settlement_time': None,
        'high_price_min': policy.state_red_high_price_min if policy.state_red_high_price_min is not None else policy.high_price_min,
        'score_price_min': policy.state_red_score_price_min,
        'score_price_max': policy.state_red_score_price_max,
        'low_score_max': policy.low_score_max,
    }
    pending: list[tuple[datetime, OfficialTrade]] = []
    for trade in sorted(rows, key=lambda r: r.dt):
        _release_pending(pending, trade.dt, state)
        blocked, _reason, _bars = _maybe_trigger(policy, state, trade.dt)
        if blocked:
            continue
        release_at = trade.dt + timedelta(minutes=15 * max(0, policy.update_lag_bars))
        pending.append((release_at, trade))
    _release_pending(pending, now_dt, state)
    blocked, reason, _bars = _maybe_trigger(policy, state, now_dt)
    return blocked, reason


def decision_to_dict(decision: RiskV2Decision) -> dict[str, Any]:
    return asdict(decision)
