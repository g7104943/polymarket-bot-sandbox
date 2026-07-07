from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PM = ROOT / 'polymarket'

MODE_SUFFIX_RE = re.compile(r'__(simulation|live|backtest)(?:_[a-z0-9_]+)?$', re.IGNORECASE)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _round(value: Any, digits: int = 4) -> float:
    try:
        return round(float(value), digits)
    except Exception:
        return 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _load_json(path: Path, default: Any = None) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return default


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding='utf-8').splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def base_of(log_dir: str) -> str:
    return MODE_SUFFIX_RE.sub('', str(log_dir or '').strip())


def mode_of(log_dir: str, fallback: str = 'simulation') -> str:
    match = MODE_SUFFIX_RE.search(str(log_dir or '').strip())
    if match:
        return str(match.group(1) or '').lower() or fallback
    return fallback


def _trade_file_candidates(log_dir: str, mode: str) -> list[Path]:
    raw_dir = PM / str(log_dir or '').strip()
    base = PM / base_of(log_dir)
    roots: list[Path] = []
    for candidate_root in (raw_dir, base):
        if candidate_root not in roots:
            roots.append(candidate_root)
    candidates: list[Path] = []
    for root in roots:
        candidates.append(root / f'prediction_trades.{mode}.json')
        if mode == 'simulation':
            candidates.append(root / 'prediction_trades.json')
    return candidates


def _ledger_file_candidates(log_dir: str, mode: str) -> list[Path]:
    raw_dir = PM / str(log_dir or '').strip()
    base = PM / base_of(log_dir)
    roots: list[Path] = []
    for candidate_root in (raw_dir, base):
        if candidate_root not in roots:
            roots.append(candidate_root)
    candidates: list[Path] = []
    for root in roots:
        candidates.append(root / f'pending_order_ledger.{mode}.jsonl')
        if mode == 'simulation':
            candidates.append(root / 'pending_order_ledger.jsonl')
    return candidates


def _event_file_candidates(log_dir: str, mode: str) -> list[Path]:
    raw_dir = PM / str(log_dir or '').strip()
    base = PM / base_of(log_dir)
    roots: list[Path] = []
    for candidate_root in (raw_dir, base):
        if candidate_root not in roots:
            roots.append(candidate_root)
    candidates: list[Path] = []
    for root in roots:
        candidates.append(root / f'execution_v2_events.{mode}.jsonl')
        if mode == 'simulation':
            candidates.append(root / 'execution_v2_events.jsonl')
    return candidates


def read_trades(log_dir: str, mode: str) -> list[dict[str, Any]]:
    for path in _trade_file_candidates(log_dir, mode):
        payload = _load_json(path, [])
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
    return []


def read_ledger(log_dir: str, mode: str) -> list[dict[str, Any]]:
    for path in _ledger_file_candidates(log_dir, mode):
        rows = _load_jsonl(path)
        if rows:
            return rows
    return []


def read_execution_v2_events(log_dir: str, mode: str) -> list[dict[str, Any]]:
    for path in _event_file_candidates(log_dir, mode):
        rows = _load_jsonl(path)
        if rows:
            return rows
    return []


def _logical_key(row: dict[str, Any]) -> str:
    for key in ('conditionId', 'marketSlug', 'id'):
        value = str(row.get(key) or '').strip()
        if value:
            return value
    return f'_no_key_{id(row)}'


def _parse_ts(value: Any) -> datetime | None:
    text = str(value or '').strip()
    if not text:
        return None
    try:
        if text.endswith('Z'):
            return datetime.fromisoformat(text.replace('Z', '+00:00')).astimezone(timezone.utc)
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def summarize_raw_simulation(rows: list[dict[str, Any]], symbol: str) -> dict[str, Any]:
    symbol_text = str(symbol or '').strip().upper()
    scoped = [row for row in rows if str(row.get('symbol') or '').strip().upper() == symbol_text]
    executed = [row for row in scoped if str(row.get('status') or '').strip().lower() == 'executed']
    by_order: dict[str, str | None] = {}
    total_pnl = 0.0
    latest_ts: datetime | None = None
    for row in scoped:
        logical_key = _logical_key(row)
        by_order.setdefault(logical_key, None)
        if str(row.get('status') or '').strip().lower() == 'executed':
            result = str(row.get('result') or '').strip().lower()
            if result in {'win', 'lose'}:
                by_order[logical_key] = result
    for row in executed:
        total_pnl += _safe_float(row.get('pnl'))
        ts = _parse_ts(row.get('timestamp'))
        if ts is not None and (latest_ts is None or ts > latest_ts):
            latest_ts = ts
    wins = sum(1 for value in by_order.values() if value == 'win')
    losses = sum(1 for value in by_order.values() if value == 'lose')
    pending = sum(1 for value in by_order.values() if value is None)
    completed = wins + losses
    return {
        'rawSimTrades': len(by_order),
        'rawSimWins': wins,
        'rawSimLosses': losses,
        'rawSimPending': pending,
        'rawSimWinRate': round((wins / completed * 100.0) if completed else 0.0, 2),
        'rawSimPnlUsd': round(total_pnl, 2),
        'rawSimFills': len(executed),
        'rawLatestTradeAt': latest_ts.isoformat() if latest_ts else None,
    }


def _trade_fill_ratio(row: dict[str, Any]) -> float:
    amount = max(_safe_float(row.get('amount')), 0.0)
    if amount <= 0.0:
        return 1.0
    eff_raw = row.get('effectiveFillableUsdAtEntry')
    queue_raw = row.get('queueCompetitionUsdAtEntry')
    if eff_raw is None and queue_raw is None:
        return 1.0
    eff = max(_safe_float(eff_raw), 0.0)
    queue = max(_safe_float(queue_raw), 0.0)
    if eff <= 0.0 and queue <= 0.0:
        return 1.0
    if eff <= 0.0:
        return 0.0
    return max(0.0, min(1.0, eff / max(amount + queue, amount)))


def _partial_fill_ratio(ledger_rows: list[dict[str, Any]]) -> float:
    by_order: dict[str, set[str]] = defaultdict(set)
    for row in ledger_rows:
        order_id = str(row.get('orderId') or '').strip()
        if not order_id:
            continue
        by_order[order_id].add(str(row.get('event') or '').strip())
    if not by_order:
        return 0.0
    partial = sum(1 for events in by_order.values() if 'partial_fill' in events)
    return round(partial / max(len(by_order), 1), 4)


def build_order_summaries(
    trade_rows: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
    symbol: str,
) -> list[dict[str, Any]]:
    symbol_text = str(symbol or '').strip().upper()
    created_by_order: dict[str, dict[str, Any]] = {}
    fill_agg: dict[str, dict[str, Any]] = defaultdict(lambda: {'filledUsd': 0.0, 'fillPriceNumer': 0.0, 'fillEvents': 0})
    trade_by_order: dict[str, dict[str, Any]] = {}

    for row in ledger_rows:
        if str(row.get('symbol') or '').strip().upper() != symbol_text:
            continue
        order_id = str(row.get('orderId') or '').strip()
        if not order_id:
            continue
        event = str(row.get('event') or '').strip().lower()
        if event == 'created' and order_id not in created_by_order:
            created_by_order[order_id] = row
        if event in {'filled', 'partial_fill'}:
            amount_usd = max(_safe_float(row.get('amountUsd')), 0.0)
            avg_fill_price = max(_safe_float(row.get('avgFillPrice')), 0.0)
            fill_agg[order_id]['filledUsd'] += amount_usd
            fill_agg[order_id]['fillPriceNumer'] += amount_usd * avg_fill_price
            fill_agg[order_id]['fillEvents'] += 1

    for row in trade_rows:
        if str(row.get('symbol') or '').strip().upper() != symbol_text:
            continue
        order_id = str(row.get('orderId') or '').strip()
        if order_id:
            trade_by_order[order_id] = row

    all_order_ids = sorted(set(created_by_order) | set(fill_agg) | set(trade_by_order))
    summaries: list[dict[str, Any]] = []
    for order_id in all_order_ids:
        created = created_by_order.get(order_id, {})
        trade = trade_by_order.get(order_id, {})
        fills = fill_agg.get(order_id, {})
        requested_usd = max(
            _safe_float(created.get('amountUsd')),
            _safe_float(trade.get('requestedAmountUsd')),
            _safe_float(trade.get('amount')),
        )
        filled_usd = max(_safe_float(fills.get('filledUsd')), _safe_float(trade.get('amount')))
        avg_fill_price = (
            _safe_float(fills.get('fillPriceNumer')) / filled_usd
            if filled_usd > 0 and _safe_float(fills.get('fillPriceNumer')) > 0
            else _safe_float(trade.get('avgActualFillPrice'))
        )
        summaries.append({
            'orderId': order_id,
            'symbol': symbol_text,
            'conditionId': str(created.get('conditionId') or trade.get('conditionId') or '').strip() or None,
            'marketSlug': str(created.get('marketSlug') or trade.get('marketSlug') or '').strip() or None,
            'direction': str(created.get('direction') or trade.get('direction') or '').strip().upper() or None,
            'confidence': _safe_float(created.get('confidence') if created.get('confidence') is not None else trade.get('confidence')),
            'limitPrice': _safe_float(created.get('limitPrice') if created.get('limitPrice') is not None else trade.get('limitPriceConfigured')),
            'bestAsk': _safe_float(created.get('bestAsk')),
            'queueCompetitionUsdAtEntry': _safe_float(created.get('queueCompetitionUsdAtEntry')),
            'effectiveFillableUsdAtEntry': _safe_float(created.get('effectiveFillableUsdAtEntry')),
            'requestedUsd': requested_usd,
            'filledUsd': filled_usd,
            'fillRatio': _clamp((filled_usd / requested_usd) if requested_usd > 0 else 0.0, 0.0, 1.0),
            'avgFillPrice': avg_fill_price,
            'result': str(trade.get('result') or '').strip().lower() or None,
            'createdAt': created.get('createdAt') or created.get('timestamp'),
            'expiresAt': created.get('expiresAt'),
            'targetPeriodEndTs': _safe_int(created.get('targetPeriodEndTs') or trade.get('targetPeriodEndTs')),
            'fillEvents': _safe_int(fills.get('fillEvents')),
        })
    return summaries


def _equivalent_row_pnl(row: dict[str, Any], assumed_price: float, fill_ratio: float) -> tuple[float, float]:
    amount = max(_safe_float(row.get('amount')), 0.0)
    fill_ratio = max(0.0, min(1.0, fill_ratio))
    filled_amount = amount * fill_ratio
    if filled_amount <= 0.0 or assumed_price <= 0.0:
        return 0.0, 0.0
    shares = filled_amount / assumed_price
    result = str(row.get('result') or '').strip().lower()
    if result == 'win':
        return shares - filled_amount, filled_amount
    if result == 'lose':
        return -filled_amount, filled_amount
    return 0.0, filled_amount


def compute_live_equivalent_metrics(
    rows: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
    symbol: str,
    default_limit_price: float | None = None,
    event_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    symbol_text = str(symbol or '').strip().upper()
    executed = [
        row for row in rows
        if str(row.get('symbol') or '').strip().upper() == symbol_text
        and str(row.get('status') or '').strip().lower() == 'executed'
    ]
    if not executed:
        return {
            'amountWeightedActualFillPrice': 0.0,
            'amountWeightedLimitPrice': _round(default_limit_price or 0.0, 4),
            'improvementVsLimit': 0.0,
            'bestOfferCoverageRatio': 0.0,
            'queueEvidenceCoverageRatio': 0.0,
            'queueEquivalentFillRatio': 0.0,
            'partialFillRatio': 0.0,
            'worstCaseLimitPnlUsd': 0.0,
            'liveEquivalentTrades': 0.0,
            'liveEquivalentWinRate': 0.0,
            'liveEquivalentPnlUsd': 0.0,
            'realismGateStatus': 'fail',
            'realismFailReasons': ['无成交样本，无法形成真实交易等价口径'],
            'primaryRealismFailReason': 'no_trade_evidence',
        }

    total_amount = sum(max(_safe_float(row.get('amount')), 0.0) for row in executed)
    actual_price_numer = 0.0
    limit_price_numer = 0.0
    best_offer_coverage = 0
    queue_coverage = 0
    queue_fill_ratio_sum = 0.0
    win_fill_ratio_sum = 0.0
    loss_fill_ratio_sum = 0.0
    win_fill_count = 0
    loss_fill_count = 0
    by_order_equivalent: dict[str, dict[str, float]] = defaultdict(lambda: {'raw_amount': 0.0, 'equiv_amount': 0.0, 'equiv_pnl': 0.0})
    worst_case_limit_pnl = 0.0

    for row in executed:
        amount = max(_safe_float(row.get('amount')), 0.0)
        actual_price = _safe_float(row.get('avgActualFillPrice') or row.get('tokenPrice') or 0.0)
        assumed_price = _safe_float(row.get('limitPriceConfigured') or default_limit_price or actual_price)
        fill_ratio = _trade_fill_ratio(row)
        actual_price_numer += amount * actual_price
        limit_price_numer += amount * assumed_price
        if row.get('bestAskAtEntry') is not None:
            best_offer_coverage += 1
        if row.get('queueCompetitionUsdAtEntry') is not None or row.get('effectiveFillableUsdAtEntry') is not None:
            queue_coverage += 1
        queue_fill_ratio_sum += fill_ratio
        result = str(row.get('result') or '').strip().lower()
        if result == 'win':
            win_fill_ratio_sum += fill_ratio
            win_fill_count += 1
        elif result == 'lose':
            loss_fill_ratio_sum += fill_ratio
            loss_fill_count += 1
        pnl_equiv, equiv_amount = _equivalent_row_pnl(row, assumed_price, fill_ratio)
        logical_key = _logical_key(row)
        by_order_equivalent[logical_key]['raw_amount'] += amount
        by_order_equivalent[logical_key]['equiv_amount'] += equiv_amount
        by_order_equivalent[logical_key]['equiv_pnl'] += pnl_equiv
        pnl_worst, _ = _equivalent_row_pnl(row, assumed_price, 1.0)
        worst_case_limit_pnl += pnl_worst

    logical_equivalent_trades = 0.0
    logical_wins = 0
    logical_losses = 0
    for agg in by_order_equivalent.values():
        raw_amount = max(float(agg.get('raw_amount') or 0.0), 0.0)
        equiv_amount = max(float(agg.get('equiv_amount') or 0.0), 0.0)
        if raw_amount > 0.0:
            logical_equivalent_trades += min(1.0, equiv_amount / raw_amount)
        pnl = float(agg.get('equiv_pnl') or 0.0)
        if pnl > 1e-9:
            logical_wins += 1
        elif pnl < -1e-9:
            logical_losses += 1
    completed = logical_wins + logical_losses

    amount_weighted_actual_fill_price = actual_price_numer / total_amount if total_amount > 0 else 0.0
    amount_weighted_limit_price = limit_price_numer / total_amount if total_amount > 0 else _safe_float(default_limit_price)
    improvement_vs_limit = amount_weighted_limit_price - amount_weighted_actual_fill_price
    best_offer_coverage_ratio = best_offer_coverage / max(len(executed), 1)
    queue_evidence_coverage_ratio = queue_coverage / max(len(executed), 1)
    queue_equivalent_fill_ratio = queue_fill_ratio_sum / max(len(executed), 1)
    partial_fill_ratio = _partial_fill_ratio(ledger_rows)
    win_fill_ratio = win_fill_ratio_sum / max(win_fill_count, 1)
    loss_fill_ratio = loss_fill_ratio_sum / max(loss_fill_count, 1)
    fill_bias_gap = loss_fill_ratio - win_fill_ratio

    reasons: list[str] = []
    primary_reason = ''
    if improvement_vs_limit > 0.10:
        reasons.append('金额加权实际成交价显著优于上限价，原始模拟收益偏乐观')
        primary_reason = primary_reason or 'fill_realism_fail'
    if best_offer_coverage_ratio < 0.75:
        reasons.append('最优卖价证据覆盖不足')
        primary_reason = primary_reason or 'fill_realism_fail'
    if queue_evidence_coverage_ratio < 0.75 or queue_equivalent_fill_ratio < 0.75:
        reasons.append('排队/可成交深度现实性不足')
        primary_reason = primary_reason or 'queue_realism_fail'
    if partial_fill_ratio > 0.25:
        reasons.append('部分成交比例偏高，真实成交比例需要更保守处理')
        primary_reason = primary_reason or 'queue_realism_fail'
    if fill_bias_gap > 0.10:
        reasons.append('赢单成交比例明显低于输单，存在方向性成交偏差')
        primary_reason = primary_reason or 'queue_realism_fail'
    if worst_case_limit_pnl <= 0.0:
        reasons.append('按上限价与保守 fill 口径重算后收益不成立')
        primary_reason = primary_reason or 'worst_case_pnl_fail'

    event_rows = [row for row in (event_rows or []) if isinstance(row, dict)]
    by_order_chased: set[str] = set()
    by_order_final: set[str] = set()
    by_order_cancel_invalid: set[str] = set()
    by_order_cancel_stale: set[str] = set()
    by_order_cancel_pre_expiry: set[str] = set()
    for row in event_rows:
        event_type = str(row.get('eventType') or '').strip().lower()
        order_id = str(row.get('orderId') or '').strip()
        if not order_id:
            continue
        if event_type == 'chase_reprice_posted':
            by_order_chased.add(order_id)
        elif event_type == 'final_aggressive_posted':
            by_order_final.add(order_id)
        elif event_type == 'cancelled_on_direction_invalidation':
            by_order_cancel_invalid.add(order_id)
        elif event_type == 'cancelled_on_stale_queue':
            by_order_cancel_stale.add(order_id)
        elif event_type == 'cancelled_pre_expiry_timeout':
            by_order_cancel_pre_expiry.add(order_id)

    order_summaries = build_order_summaries(rows, ledger_rows, symbol)
    chase_recovery_usd = 0.0
    for summary in order_summaries:
        if str(summary.get('orderId') or '') in by_order_chased:
            chase_recovery_usd += max(_safe_float(summary.get('filledUsd')) - _safe_float(summary.get('requestedUsd')) * 0.5, 0.0)

    if event_rows:
        replay_realism_verdict = 'event_replay_base_with_execution_v2_event_evidence'
    elif order_summaries:
        replay_realism_verdict = 'event_replay_base_calibrated_from_order_lifecycle'
    else:
        replay_realism_verdict = 'event_replay_base_no_order_lifecycle_evidence'

    return {
        'amountWeightedActualFillPrice': round(amount_weighted_actual_fill_price, 4),
        'amountWeightedLimitPrice': round(amount_weighted_limit_price, 4),
        'improvementVsLimit': round(improvement_vs_limit, 4),
        'bestOfferCoverageRatio': round(best_offer_coverage_ratio, 4),
        'queueEvidenceCoverageRatio': round(queue_evidence_coverage_ratio, 4),
        'queueEquivalentFillRatio': round(queue_equivalent_fill_ratio, 4),
        'partialFillRatio': round(partial_fill_ratio, 4),
        'winFillRatio': round(win_fill_ratio, 4),
        'lossFillRatio': round(loss_fill_ratio, 4),
        'fillBiasGap': round(fill_bias_gap, 4),
        'worstCaseLimitPnlUsd': round(worst_case_limit_pnl, 2),
        'liveEquivalentTrades': round(logical_equivalent_trades, 2),
        'liveEquivalentWinRate': round((logical_wins / completed * 100.0) if completed else 0.0, 2),
        'liveEquivalentPnlUsd': round(sum(float(agg.get('equiv_pnl') or 0.0) for agg in by_order_equivalent.values()), 2),
        'chaseRecoveryUsd': round(chase_recovery_usd, 2),
        'chaseTriggeredTrades': len(by_order_chased),
        'finalAggressiveTriggeredTrades': len(by_order_final),
        'cancelledOnInvalidationTrades': len(by_order_cancel_invalid),
        'cancelledOnStaleQueueTrades': len(by_order_cancel_stale),
        'cancelledPreExpiryTimeoutTrades': len(by_order_cancel_pre_expiry),
        'replayRealismVerdict': replay_realism_verdict,
        'realismGateStatus': 'pass' if not reasons else 'fail',
        'realismFailReasons': reasons,
        'primaryRealismFailReason': primary_reason or 'pass',
    }


def compute_realistic_simulation_metrics(
    raw: dict[str, Any],
    live_equivalent: dict[str, Any],
) -> dict[str, Any]:
    """Blend raw simulation and execution evidence into a display-grade simulation PnL.

    The old live-equivalent number is intentionally conservative. This layer keeps it
    as a stress reference, but uses order-book evidence to stay closer to the raw
    simulation when the execution evidence is good.
    """
    raw_pnl = _safe_float(raw.get('rawSimPnlUsd'))
    conservative_pnl = _safe_float(live_equivalent.get('liveEquivalentPnlUsd'), raw_pnl)
    raw_trades = _safe_int(raw.get('rawSimTrades'))
    raw_win_rate = _safe_float(raw.get('rawSimWinRate'))
    best_offer = _clamp(_safe_float(live_equivalent.get('bestOfferCoverageRatio')), 0.0, 1.0)
    queue_evidence = _clamp(_safe_float(live_equivalent.get('queueEvidenceCoverageRatio')), 0.0, 1.0)
    queue_fill = _clamp(_safe_float(live_equivalent.get('queueEquivalentFillRatio')), 0.0, 1.0)
    partial_fill = _clamp(_safe_float(live_equivalent.get('partialFillRatio')), 0.0, 1.0)
    fill_bias_gap = _safe_float(live_equivalent.get('fillBiasGap'))
    improvement_vs_limit = _safe_float(live_equivalent.get('improvementVsLimit'))

    execution_evidence = _clamp((0.45 * best_offer) + (0.35 * queue_evidence) + (0.20 * queue_fill), 0.0, 1.0)
    partial_uncertainty = max(0.0, partial_fill - 0.75)
    raw_weight = _clamp(0.62 + (0.28 * execution_evidence) - (0.12 * partial_uncertainty), 0.55, 0.90)

    if raw_trades <= 0:
        realistic_pnl = 0.0
        display_policy = '无成交样本，拟真盈亏按0处理。'
    elif conservative_pnl >= raw_pnl:
        # When the limit-price replay is better than raw, only accept most of that
        # improvement if the order-book evidence is strong.
        improvement_weight = _clamp(0.35 + (0.45 * execution_evidence), 0.35, 0.85)
        realistic_pnl = raw_pnl + ((conservative_pnl - raw_pnl) * improvement_weight)
        display_policy = '成交证据支持时，允许使用一部分真实成交价改善。'
    else:
        realistic_pnl = (raw_pnl * raw_weight) + (conservative_pnl * (1.0 - raw_weight))
        display_policy = '订单薄证据越充分越靠近原始模拟，证据越弱越向压力口径收敛。'

    # Keep the new default close to real execution without turning it into a
    # hand-waved upside multiplier.
    if raw_trades > 0 and conservative_pnl < raw_pnl:
        optimistic_cap = raw_pnl + max(0.0, abs(raw_pnl - conservative_pnl) * 0.03)
        realistic_pnl = min(realistic_pnl, optimistic_cap)
    elif raw_trades > 0 and conservative_pnl >= raw_pnl and improvement_vs_limit <= 0.0:
        realistic_pnl = min(realistic_pnl, conservative_pnl)

    sample_too_small = raw_trades < 20
    return {
        'realisticSimTrades': raw_trades,
        'realisticSimWinRate': round(raw_win_rate, 2),
        'realisticSimPnlUsd': round(realistic_pnl, 2),
        'conservativeStressPnlUsd': round(conservative_pnl, 2),
        'realisticExecutionEvidenceScore': round(execution_evidence, 4),
        'realisticRawBlendWeight': round(raw_weight, 4),
        'fillBiasGap': round(fill_bias_gap, 4),
        'realisticSampleTooSmallForLiveCandidate': sample_too_small,
        'realisticPnlBasis': 'realistic_simulated_pnl',
        'realisticModelPolicyChinese': display_policy,
    }


def row_scope(row: dict[str, Any]) -> str:
    return 'lowprice' if str(row.get('_config_set') or '').strip() == 'lowprice' else 'all'
