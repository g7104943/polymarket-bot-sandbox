#!/usr/bin/env python3
"""
诊断脚本：对比验证 模型预测 vs Binance 实际涨跌 vs Polymarket 市场结算

目的：
  - 验证模型预测方向是否与 Binance 实际涨跌一致
  - 验证 Polymarket 市场结算是否与 Binance 一致
  - 量化模拟K线 close 与实际 close 的偏差
  - 对比 Binance 价格与 Chainlink 价格的差异

使用:
  # 分析最近 N 个 15m 周期 (默认最近 48 个 = 12 小时)
  python scripts/diagnose_prediction_alignment.py

  # 分析特定时间段
  python scripts/diagnose_prediction_alignment.py --hours 24

  # 实时监控模式：等待下一个周期结束后自动验证
  python scripts/diagnose_prediction_alignment.py --live
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ─── 配置 ─────────────────────────────────────────────────
ASSETS = ["BTC_USDT", "ETH_USDT", "XRP_USDT"]
ASSET_TO_BINANCE = {
    "BTC_USDT": "BTCUSDT",
    "ETH_USDT": "ETHUSDT",
    "XRP_USDT": "XRPUSDT",
}
SLUG_MAP = {"BTC_USDT": "btc", "ETH_USDT": "eth", "XRP_USDT": "xrp"}
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
BINANCE_API = "https://api.binance.com"
CHAINLINK_DATA_API = "https://data.chain.link/streams"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Accept": "application/json",
}

PREDICTION_FILES = [
    PROJECT_ROOT / "polymarket" / "predictions_v5_exp8.json",
    PROJECT_ROOT / "polymarket" / "predictions_v5_exp9.json",
    PROJECT_ROOT / "polymarket" / "predictions.json",
]

TRADE_LOG_DIRS = sorted(
    (PROJECT_ROOT / "polymarket").glob("logs_v5_exp*_*/"),
)

OUTPUT_DIR = PROJECT_ROOT / "logs" / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  1. 数据获取
# ═══════════════════════════════════════════════════════════════

def fetch_binance_kline(symbol: str, start_ts: int, end_ts: int,
                        interval: str = "15m") -> Optional[Dict]:
    """从 Binance 获取指定时间段的 K线数据。

    Args:
        symbol: Binance 符号 (如 "BTCUSDT")
        start_ts: 开始时间戳 (Unix 秒)
        end_ts: 结束时间戳 (Unix 秒)
        interval: K线周期

    Returns:
        {"open": float, "high": float, "low": float, "close": float, "volume": float}
        或 None (获取失败)
    """
    url = f"{BINANCE_API}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ts * 1000,
        "endTime": end_ts * 1000 - 1,  # Binance: endTime 是闭区间
        "limit": 1,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        k = data[0]
        return {
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "open_time_ms": int(k[0]),
            "close_time_ms": int(k[6]),
        }
    except Exception as e:
        print(f"  [错误] Binance K线获取失败 ({symbol}): {e}")
        return None


def fetch_binance_1m_klines(symbol: str, start_ts: int, count: int = 15
                            ) -> List[Dict]:
    """获取多根 1m K线，用于重建模拟 K线。"""
    url = f"{BINANCE_API}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": "1m",
        "startTime": start_ts * 1000,
        "limit": count,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        return [
            {
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "open_time_ms": int(k[0]),
            }
            for k in data
        ]
    except Exception as e:
        print(f"  [错误] Binance 1m K线获取失败 ({symbol}): {e}")
        return []


def fetch_polymarket_result(slug: str) -> Optional[Dict]:
    """获取 Polymarket 市场结算结果。

    Returns:
        {"resolved": bool, "winner": "Up"/"Down"/None, "description": str,
         "outcomes": list, "outcome_prices": list}
    """
    url = f"{GAMMA_API}/events/slug/{slug}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 404:
            return {"resolved": False, "winner": None, "error": "市场不存在"}
        r.raise_for_status()
        event = r.json()
        if not event.get("markets"):
            return {"resolved": False, "winner": None, "error": "无市场数据"}

        market = event["markets"][0]
        outcomes_raw = market.get("outcomes", '["Up","Down"]')
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        prices_raw = market.get("outcomePrices", "[]")
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw

        closed = market.get("closed", False) or market.get("archived", False)
        description = market.get("description", "")

        # 判断赢家
        winner = None
        resolved = False
        if prices and len(prices) >= 2:
            p0 = float(prices[0])
            p1 = float(prices[1])
            if p0 >= 0.99 and p1 <= 0.01:
                resolved = True
                winner = outcomes[0] if outcomes else "Up"
            elif p1 >= 0.99 and p0 <= 0.01:
                resolved = True
                winner = outcomes[1] if outcomes else "Down"

        # 也尝试 CLOB API
        if not resolved:
            cond_id = market.get("conditionId", "")
            if cond_id:
                clob_result = _fetch_clob_result(cond_id)
                if clob_result and clob_result.get("resolved"):
                    resolved = True
                    winner = clob_result["winner"]

        return {
            "resolved": resolved,
            "winner": winner,
            "closed": closed,
            "description": description,
            "outcomes": outcomes,
            "outcome_prices": [float(p) for p in prices] if prices else [],
        }
    except Exception as e:
        return {"resolved": False, "winner": None, "error": str(e)}


def _fetch_clob_result(condition_id: str) -> Optional[Dict]:
    """从 CLOB API 获取结算结果。"""
    try:
        url = f"{CLOB_API}/markets/{condition_id}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        m = r.json()
        tokens = m.get("tokens", [])
        winner_token = next((t for t in tokens if t.get("winner")), None)
        if winner_token:
            outcome = str(winner_token.get("outcome", ""))
            winner = "Down" if outcome.lower() == "down" else "Up"
            return {"resolved": True, "winner": winner}
        return None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
#  2. 核心分析
# ═══════════════════════════════════════════════════════════════

def analyze_period(period_start_ts: int, verbose: bool = True) -> List[Dict]:
    """分析单个 15m 周期: 模型预测 vs Binance 实际 vs Polymarket 结算。

    Args:
        period_start_ts: 周期起始 Unix 时间戳 (秒)

    Returns:
        每个资产一条记录的列表
    """
    period_end_ts = period_start_ts + 900  # 15m = 900s
    prev_bar_start = period_start_ts - 900

    results = []

    for asset in ASSETS:
        symbol_bn = ASSET_TO_BINANCE[asset]
        slug_sym = SLUG_MAP[asset]
        slug = f"{slug_sym}-updown-15m-{period_start_ts}"

        record: Dict[str, Any] = {
            "asset": asset,
            "period_start_ts": period_start_ts,
            "period_start_utc": datetime.fromtimestamp(period_start_ts, tz=timezone.utc).isoformat(),
            "slug": slug,
        }

        # ─── A. Binance 实际 OHLCV ───
        # 目标周期的 15m K线
        target_kline = fetch_binance_kline(symbol_bn, period_start_ts, period_end_ts)
        if target_kline:
            record["binance_target_open"] = target_kline["open"]
            record["binance_target_close"] = target_kline["close"]
            record["binance_target_high"] = target_kline["high"]
            record["binance_target_low"] = target_kline["low"]
            # Polymarket 判定: close >= open 则 "Up"
            record["binance_close_vs_open"] = "Up" if target_kline["close"] >= target_kline["open"] else "Down"
        else:
            record["binance_target_open"] = None
            record["binance_target_close"] = None
            record["binance_close_vs_open"] = None

        # 前一根 15m K线 (用于验证模型标签定义)
        prev_kline = fetch_binance_kline(symbol_bn, prev_bar_start, period_start_ts)
        if prev_kline and target_kline:
            record["binance_prev_close"] = prev_kline["close"]
            # 模型标签定义: close[t+1] > close[t]
            record["binance_next_close_gt_prev_close"] = (
                "Up" if target_kline["close"] > prev_kline["close"] else "Down"
            )
        else:
            record["binance_prev_close"] = None
            record["binance_next_close_gt_prev_close"] = None

        # ─── B. 模拟 K线 (T-120s) ───
        # 重建: 取前一根 bar 的前 13 根 1m K线
        sim_1m = fetch_binance_1m_klines(symbol_bn, prev_bar_start, 15)
        if sim_1m and len(sim_1m) >= 3:
            # T-120s 时只有前 13 根 (覆盖 0~12 分钟)
            sim_bars = sim_1m[:13]  # 前 13 分钟
            sim_close = sim_bars[-1]["close"]
            record["simulated_close_at_t120s"] = sim_close
            actual_close = prev_kline["close"] if prev_kline else None
            if actual_close:
                record["sim_vs_actual_close_diff"] = sim_close - actual_close
                record["sim_vs_actual_close_pct"] = (
                    (sim_close - actual_close) / actual_close * 100
                )
                # 用模拟 close 做参考, 判断方向
                if target_kline:
                    record["model_label_with_sim_close"] = (
                        "Up" if target_kline["close"] > sim_close else "Down"
                    )
        else:
            record["simulated_close_at_t120s"] = None

        # ─── C. Polymarket 结算 ───
        pm_result = fetch_polymarket_result(slug)
        if pm_result:
            record["pm_resolved"] = pm_result.get("resolved", False)
            record["pm_winner"] = pm_result.get("winner")
            record["pm_description"] = pm_result.get("description", "")[:100]
        else:
            record["pm_resolved"] = False
            record["pm_winner"] = None

        # ─── D. 对齐检查 ───
        if record.get("binance_close_vs_open") and record.get("pm_winner"):
            record["binance_vs_pm_match"] = (
                record["binance_close_vs_open"] == record["pm_winner"]
            )
        else:
            record["binance_vs_pm_match"] = None

        if (record.get("binance_next_close_gt_prev_close") and
                record.get("binance_close_vs_open")):
            record["label_vs_market_match"] = (
                record["binance_next_close_gt_prev_close"] ==
                record["binance_close_vs_open"]
            )
        else:
            record["label_vs_market_match"] = None

        results.append(record)

    return results


def analyze_trade_logs() -> List[Dict]:
    """分析所有 exp8/exp9 的交易日志, 获取历史交易的预测方向和结果。"""
    trades = []
    for log_dir in TRADE_LOG_DIRS:
        trades_file = log_dir / "prediction_trades.json"
        if not trades_file.exists():
            continue
        try:
            with open(trades_file) as f:
                data = json.load(f)
            if not isinstance(data, list):
                continue
            for t in data:
                if t.get("status") != "executed":
                    continue
                # 提取 period_start_ts 从 marketSlug
                slug = t.get("marketSlug", "")
                parts = slug.rsplit("-", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    period_start_ts = int(parts[1])
                else:
                    continue
                trades.append({
                    "id": t.get("id"),
                    "log_dir": log_dir.name,
                    "symbol": t.get("symbol"),
                    "direction": t.get("direction"),
                    "confidence": t.get("confidence"),
                    "amount": t.get("amount"),
                    "tokenOutcome": t.get("tokenOutcome"),
                    "tokenPrice": t.get("tokenPrice"),
                    "marketSlug": slug,
                    "period_start_ts": period_start_ts,
                    "result": t.get("result"),
                    "pnl": t.get("pnl"),
                })
        except (json.JSONDecodeError, KeyError):
            continue
    return trades


# ═══════════════════════════════════════════════════════════════
#  3. 报告生成
# ═══════════════════════════════════════════════════════════════

def print_period_analysis(records: List[Dict]):
    """打印单个周期的分析结果。"""
    if not records:
        return

    ts = records[0]["period_start_ts"]
    utc_str = records[0]["period_start_utc"]
    print(f"\n{'='*70}")
    print(f"  周期: {utc_str} (ts={ts})")
    print(f"{'='*70}")

    for r in records:
        asset = r["asset"]
        print(f"\n  [{asset}]")

        # Binance 数据
        if r.get("binance_target_open"):
            print(f"    Binance 目标K线: O={r['binance_target_open']:.4f} "
                  f"C={r['binance_target_close']:.4f}")
            print(f"    Binance close>=open 判定: {r['binance_close_vs_open']}")

        if r.get("binance_prev_close"):
            print(f"    Binance 前一K线close: {r['binance_prev_close']:.4f}")
            print(f"    模型标签判定 (next_close>prev_close): "
                  f"{r['binance_next_close_gt_prev_close']}")

        if r.get("simulated_close_at_t120s"):
            print(f"    模拟K线close (T-120s): {r['simulated_close_at_t120s']:.4f}")
            if r.get("sim_vs_actual_close_diff") is not None:
                print(f"    模拟 vs 实际 close 偏差: "
                      f"{r['sim_vs_actual_close_diff']:+.4f} "
                      f"({r['sim_vs_actual_close_pct']:+.4f}%)")
            if r.get("model_label_with_sim_close"):
                print(f"    模型视角判定 (target_close>sim_close): "
                      f"{r['model_label_with_sim_close']}")

        # Polymarket 结算
        if r.get("pm_resolved"):
            print(f"    Polymarket 结算: {r['pm_winner']}")
        else:
            print(f"    Polymarket: 未结算")

        # 对齐检查
        if r.get("binance_vs_pm_match") is not None:
            match_str = "一致 ✓" if r["binance_vs_pm_match"] else "不一致 ✗"
            print(f"    Binance(close>=open) vs Polymarket: {match_str}")

        if r.get("label_vs_market_match") is not None:
            match_str = "一致 ✓" if r["label_vs_market_match"] else "不一致 ✗"
            print(f"    模型标签 vs Polymarket: {match_str}")


def generate_summary(all_records: List[Dict]) -> Dict:
    """生成汇总统计。"""
    resolved = [r for r in all_records if r.get("pm_resolved")]

    summary = {
        "total_periods_analyzed": len(all_records) // 3 if all_records else 0,
        "total_records": len(all_records),
        "resolved_records": len(resolved),
    }

    if not resolved:
        return summary

    # Binance close>=open vs Polymarket 一致率
    bn_pm_matches = [r for r in resolved if r.get("binance_vs_pm_match") is not None]
    if bn_pm_matches:
        match_count = sum(1 for r in bn_pm_matches if r["binance_vs_pm_match"])
        summary["binance_vs_pm_match_rate"] = match_count / len(bn_pm_matches) * 100
        summary["binance_vs_pm_match_count"] = f"{match_count}/{len(bn_pm_matches)}"

    # 模型标签 vs 市场解析 一致率
    label_matches = [r for r in resolved if r.get("label_vs_market_match") is not None]
    if label_matches:
        match_count = sum(1 for r in label_matches if r["label_vs_market_match"])
        summary["label_vs_market_match_rate"] = match_count / len(label_matches) * 100
        summary["label_vs_market_match_count"] = f"{match_count}/{len(label_matches)}"

    # 模拟 K线 close 偏差统计
    sim_diffs = [r["sim_vs_actual_close_pct"] for r in all_records
                 if r.get("sim_vs_actual_close_pct") is not None]
    if sim_diffs:
        summary["sim_close_pct_diff_mean"] = np.mean(sim_diffs)
        summary["sim_close_pct_diff_std"] = np.std(sim_diffs)
        summary["sim_close_pct_diff_abs_mean"] = np.mean(np.abs(sim_diffs))

    # 模拟close视角 vs 市场解析 的不一致情况
    sim_label_records = [r for r in resolved
                         if r.get("model_label_with_sim_close") and r.get("pm_winner")]
    if sim_label_records:
        sim_match = sum(
            1 for r in sim_label_records
            if r["model_label_with_sim_close"] == r["pm_winner"]
        )
        summary["sim_label_vs_pm_match_rate"] = sim_match / len(sim_label_records) * 100
        summary["sim_label_vs_pm_match_count"] = f"{sim_match}/{len(sim_label_records)}"

    # Per-asset 统计
    for asset in ASSETS:
        asset_records = [r for r in resolved if r["asset"] == asset]
        if not asset_records:
            continue
        key = asset.replace("_USDT", "")
        bn_pm = [r for r in asset_records if r.get("binance_vs_pm_match") is not None]
        if bn_pm:
            mc = sum(1 for r in bn_pm if r["binance_vs_pm_match"])
            summary[f"{key}_binance_vs_pm"] = f"{mc}/{len(bn_pm)} ({mc/len(bn_pm)*100:.1f}%)"

    return summary


# ═══════════════════════════════════════════════════════════════
#  4. 主流程
# ═══════════════════════════════════════════════════════════════

def run_retrospective(hours: int = 12, verbose: bool = True):
    """回顾分析最近 N 小时的 15m 周期。"""
    now_ts = int(time.time())
    # 对齐到 15m 边界
    current_period = (now_ts // 900) * 900
    periods_to_check = hours * 4  # 每小时 4 个 15m 周期

    print(f"\n{'#'*70}")
    print(f"  诊断: 回顾分析最近 {hours} 小时 ({periods_to_check} 个周期)")
    print(f"  当前时间: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'#'*70}")

    all_records = []

    for i in range(periods_to_check):
        period_ts = current_period - (i + 1) * 900  # 从最近的已完成周期开始
        records = analyze_period(period_ts, verbose=verbose)
        all_records.extend(records)

        if verbose:
            print_period_analysis(records)

        # 控制 API 请求频率
        time.sleep(0.3)

    # 汇总
    summary = generate_summary(all_records)

    print(f"\n{'='*70}")
    print(f"  汇总统计")
    print(f"{'='*70}")
    print(f"  分析周期数:     {summary.get('total_periods_analyzed', 0)}")
    print(f"  已结算记录数:   {summary.get('resolved_records', 0)}")

    if "binance_vs_pm_match_rate" in summary:
        print(f"\n  --- 价格源对齐 ---")
        print(f"  Binance(close>=open) vs Polymarket 一致率: "
              f"{summary['binance_vs_pm_match_rate']:.1f}% "
              f"({summary['binance_vs_pm_match_count']})")

    if "label_vs_market_match_rate" in summary:
        print(f"\n  --- 标签定义对齐 ---")
        print(f"  模型标签(next_close>prev_close) vs Polymarket 一致率: "
              f"{summary['label_vs_market_match_rate']:.1f}% "
              f"({summary['label_vs_market_match_count']})")

    if "sim_label_vs_pm_match_rate" in summary:
        print(f"  模拟close视角(target_close>sim_close) vs Polymarket 一致率: "
              f"{summary['sim_label_vs_pm_match_rate']:.1f}% "
              f"({summary['sim_label_vs_pm_match_count']})")

    if "sim_close_pct_diff_abs_mean" in summary:
        print(f"\n  --- 模拟K线偏差 ---")
        print(f"  模拟close vs 实际close 平均绝对偏差: "
              f"{summary['sim_close_pct_diff_abs_mean']:.4f}%")
        print(f"  模拟close vs 实际close 偏差均值:     "
              f"{summary['sim_close_pct_diff_mean']:+.4f}%")
        print(f"  模拟close vs 实际close 偏差标准差:   "
              f"{summary['sim_close_pct_diff_std']:.4f}%")

    for asset in ASSETS:
        key = asset.replace("_USDT", "")
        if f"{key}_binance_vs_pm" in summary:
            print(f"  {key} Binance vs PM: {summary[f'{key}_binance_vs_pm']}")

    # 保存结果
    output_file = OUTPUT_DIR / f"diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump({"summary": summary, "records": all_records}, f,
                  indent=2, default=str)
    print(f"\n  结果已保存: {output_file}")

    return summary, all_records


def run_live_monitor():
    """实时监控模式: 等待预测周期结束, 自动验证。"""
    print("\n实时监控模式启动...")
    print("等待下一个 15m 周期结束后自动验证...\n")

    while True:
        now_ts = int(time.time())
        current_period_start = (now_ts // 900) * 900
        next_period_end = current_period_start + 900

        # 等待当前周期结束 + 60s (确保市场结算)
        wait_seconds = next_period_end - now_ts + 60
        if wait_seconds > 0:
            print(f"  等待 {wait_seconds}s 直到周期 {current_period_start} 结算...")
            time.sleep(wait_seconds)

        # 分析刚刚结束的周期
        print(f"\n  分析周期: {current_period_start}")
        records = analyze_period(current_period_start)
        print_period_analysis(records)

        # 检查并读取最新的预测文件
        for pred_file in PREDICTION_FILES:
            if pred_file.exists():
                try:
                    with open(pred_file) as f:
                        pred = json.load(f)
                    pred_ts = pred.get("target_period_end_ts")
                    if pred_ts == current_period_start:
                        print(f"\n  === 预测文件匹配 ({pred_file.name}) ===")
                        for key, p in pred.get("predictions", {}).items():
                            direction = p.get("direction")
                            confidence = p.get("confidence")
                            proba_up = p.get("details", {}).get("proba_up")
                            symbol = p.get("symbol", key)

                            # 找到对应的 Binance/PM 记录
                            matching = [r for r in records
                                        if symbol.replace("/", "_") == r["asset"]]
                            if matching:
                                r = matching[0]
                                pm_winner = r.get("pm_winner", "?")
                                bn_direction = r.get("binance_close_vs_open", "?")
                                model_correct_vs_pm = (
                                    direction.upper() == pm_winner.upper()
                                    if pm_winner else "?"
                                )
                                model_correct_vs_bn = (
                                    direction.upper() == bn_direction.upper()
                                    if bn_direction else "?"
                                )
                                print(f"    {symbol}: 预测={direction} "
                                      f"(conf={confidence:.3f}, p_up={proba_up:.4f}) "
                                      f"| PM={pm_winner} "
                                      f"| BN={bn_direction} "
                                      f"| 模型vs PM: {'✓' if model_correct_vs_pm is True else '✗' if model_correct_vs_pm is False else '?'} "
                                      f"| 模型vs BN: {'✓' if model_correct_vs_bn is True else '✗' if model_correct_vs_bn is False else '?'}")
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"  预测文件读取失败: {e}")


def run_trade_log_analysis():
    """分析现有交易日志中的交易, 验证每笔交易的预测正确性。"""
    print(f"\n{'#'*70}")
    print(f"  诊断: 交易日志分析")
    print(f"{'#'*70}")

    trades = analyze_trade_logs()
    if not trades:
        print("  未找到交易记录")
        return

    # 按 period_start_ts 分组
    periods = sorted(set(t["period_start_ts"] for t in trades))
    print(f"  发现 {len(trades)} 笔交易, 涉及 {len(periods)} 个周期")

    results = []
    for period_ts in periods:
        period_trades = [t for t in trades if t["period_start_ts"] == period_ts]

        # 获取 Binance 和 PM 数据
        records = analyze_period(period_ts, verbose=False)
        time.sleep(0.3)

        for trade in period_trades:
            asset_key = f"{trade['symbol']}_USDT"
            matching = [r for r in records if r["asset"] == asset_key]
            if not matching:
                continue
            r = matching[0]

            result = {
                **trade,
                "pm_winner": r.get("pm_winner"),
                "binance_close_vs_open": r.get("binance_close_vs_open"),
                "binance_next_close_gt_prev_close": r.get("binance_next_close_gt_prev_close"),
                "model_label_with_sim_close": r.get("model_label_with_sim_close"),
            }

            # 模型预测 vs 各标准 (统一大小写比较)
            pred_dir = (trade.get("direction") or "").upper()
            if result.get("pm_winner"):
                result["pred_vs_pm"] = pred_dir == result["pm_winner"].upper()
            if result.get("binance_close_vs_open"):
                result["pred_vs_bn_close_open"] = (
                    pred_dir == result["binance_close_vs_open"].upper()
                )
            if result.get("binance_next_close_gt_prev_close"):
                result["pred_vs_bn_label"] = (
                    pred_dir == result["binance_next_close_gt_prev_close"].upper()
                )

            results.append(result)

    # 打印结果
    print(f"\n  {'─'*70}")
    print(f"  {'交易ID':<35} {'方向':<6} {'vs PM':<8} "
          f"{'vs BN(c>=o)':<12} {'vs 模型标签':<12} {'交易结果':<8}")
    print(f"  {'─'*70}")

    for r in results:
        pm_str = "✓" if r.get("pred_vs_pm") is True else (
            "✗" if r.get("pred_vs_pm") is False else "?"
        )
        bn_co_str = "✓" if r.get("pred_vs_bn_close_open") is True else (
            "✗" if r.get("pred_vs_bn_close_open") is False else "?"
        )
        bn_label_str = "✓" if r.get("pred_vs_bn_label") is True else (
            "✗" if r.get("pred_vs_bn_label") is False else "?"
        )
        trade_result = str(r.get("result") or "?")

        print(f"  {r['id']:<35} {r['direction'] or '?':<6} {pm_str:<8} "
              f"{bn_co_str:<12} {bn_label_str:<12} {trade_result:<8}")

    # 汇总
    pm_results = [r for r in results if r.get("pred_vs_pm") is not None]
    bn_co_results = [r for r in results if r.get("pred_vs_bn_close_open") is not None]
    bn_label_results = [r for r in results if r.get("pred_vs_bn_label") is not None]

    print(f"\n  {'─'*70}")
    if pm_results:
        correct = sum(1 for r in pm_results if r["pred_vs_pm"])
        print(f"  模型 vs Polymarket 正确率: {correct}/{len(pm_results)} "
              f"({correct/len(pm_results)*100:.1f}%)")
    if bn_co_results:
        correct = sum(1 for r in bn_co_results if r["pred_vs_bn_close_open"])
        print(f"  模型 vs Binance(close>=open) 正确率: {correct}/{len(bn_co_results)} "
              f"({correct/len(bn_co_results)*100:.1f}%)")
    if bn_label_results:
        correct = sum(1 for r in bn_label_results if r["pred_vs_bn_label"])
        print(f"  模型 vs 训练标签(next_close>prev_close) 正确率: "
              f"{correct}/{len(bn_label_results)} "
              f"({correct/len(bn_label_results)*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════
#  5. 入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="诊断预测对齐: 模型 vs Binance vs Polymarket"
    )
    parser.add_argument("--hours", type=int, default=12,
                        help="回顾分析的小时数 (默认 12)")
    parser.add_argument("--live", action="store_true",
                        help="实时监控模式")
    parser.add_argument("--trades", action="store_true",
                        help="分析现有交易日志")
    parser.add_argument("--quiet", action="store_true",
                        help="简洁输出 (只打印汇总)")
    args = parser.parse_args()

    if args.live:
        run_live_monitor()
    elif args.trades:
        run_trade_log_analysis()
    else:
        run_retrospective(hours=args.hours, verbose=not args.quiet)


if __name__ == "__main__":
    main()
