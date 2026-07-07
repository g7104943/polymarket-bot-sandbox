#!/usr/bin/env python3
"""
Post-only maker order viability analysis for Polymarket 15m crypto prediction markets.
══════════════════════════════════════════════════════════════════════════════
Compares:
  TAKER: GTC limit orders that cross the spread when best_ask <= limit_price (~1.5% fee).
  MAKER: Post-only limit orders that rest on the book; fill if price drops to
         limit_price within the 15-min bar.

Maker fill判定:
  - 优先使用 1 分钟 intra-bar low (data/polymarket_1m_intrabar_eth_usdt.parquet)
    → 如果 bar 内最低价 <= limit_price → maker 成交（精度 ~15x 高于快照）
  - 回退: 用下一个 bar 的 raw_p_last 近似

Uses GRU_ETH_54 parameters (prob_threshold=0.52, LIMIT_PRICE 0.45–0.56).
Reuses caching from rolling_backtest_limit_price.py.

Run: python3 polymarket/postonly_maker_analysis.py
     python3 polymarket/postonly_maker_analysis.py --no-cache
     python3 polymarket/postonly_maker_analysis.py --no-intrabar   # 强制用旧的快照模式
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse paths and load logic from rolling backtest
POLY_PROB_PATH = PROJECT_ROOT / "data" / "sentiment" / "polymarket_prob_eth_usdt.parquet"
INTRABAR_PATH = PROJECT_ROOT / "data" / "polymarket_1m_intrabar_eth_usdt.parquet"
CACHE_PATH = PROJECT_ROOT / "data" / "_cache_rolling_bt_eth.parquet"
DATA_SRC = PROJECT_ROOT / "data" / "raw"
POLY_START = "2025-10-08"
POLY_END = "2026-02-16"

# GRU_ETH_54 test case (user-specified: prob_threshold=0.52, limit 0.45–0.56)
PROB_THRESHOLD = 0.52
CAPITAL = 400
BET_PCT = 0.05
BET = CAPITAL * BET_PCT  # $20
LIMIT_LO, LIMIT_HI, LIMIT_STEP = 0.45, 0.56, 0.01

# Fees
FEE_TAKER_PCT = 0.0156   # ~1.5% taker fee
REBATE_MAKER_PCT = 0.002 # 0.2% maker rebate (effective cost = -0.2%)
COOLDOWN_BARS = 1


def load_predictions_and_prices(use_cache: bool = True, use_intrabar: bool = True) -> pd.DataFrame:
    """Load merged GRU predictions + Polymarket raw_p_last + intra-bar low/high."""
    if use_cache and CACHE_PATH.exists():
        merged = pd.read_parquet(CACHE_PATH)
        merged = merged.sort_values("ts_s").reset_index(drop=True)
        # Next-bar price: fallback maker fill proxy
        merged["raw_p_last_next"] = merged["raw_p_last"].shift(-1)
        merged = merged.iloc[:-1].copy()
        print(f"Loaded from cache: {len(merged)} rows")
    else:
        import time
        import torch
        from scripts.backtest_gru_regime import get_backtest_df_one_asset

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print("Generating GRU ETH predictions...")
        t0 = time.time()
        pred_df = get_backtest_df_one_asset(
            asset="ETH_USDT",
            test_days=365,
            device=device,
            data_src=DATA_SRC,
            after_date=POLY_START,
            end_date=POLY_END,
        )
        if "timestamp" in pred_df.columns:
            pred_df["ts_s"] = (pred_df["timestamp"] // 1000).astype(np.int64)
        else:
            pred_df["ts_s"] = (pd.to_datetime(pred_df["date"]).astype("int64") // 10**9).astype(np.int64)

        poly = pd.read_parquet(POLY_PROB_PATH)
        poly["timestamp_s"] = poly["timestamp_s"].astype(np.int64)
        merged = pred_df.merge(
            poly[["timestamp_s", "raw_p_last"]],
            left_on="ts_s",
            right_on="timestamp_s",
            how="inner",
        )
        merged = merged.sort_values("ts_s").reset_index(drop=True)
        merged["raw_p_last_next"] = merged["raw_p_last"].shift(-1)
        merged = merged.iloc[:-1]
        if "date" not in merged.columns:
            merged["date"] = pd.to_datetime(merged["ts_s"], unit="s", utc=True)
        else:
            merged["date"] = pd.to_datetime(merged["date"], utc=True)

        cols = ["ts_s", "date", "pred_prob", "pred_up", "actual_up", "raw_p_last", "raw_p_last_next"]
        cols = [c for c in cols if c in merged.columns]
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        merged[cols].to_parquet(CACHE_PATH, index=False)
        print(f"Merged and cached: {len(merged)} rows ({time.time()-t0:.1f}s)")
        merged = merged[cols]

    # ── 合并 intra-bar low/high 数据（如果有） ──
    merged["has_intrabar"] = False
    merged["intrabar_low"] = np.nan
    merged["intrabar_high"] = np.nan

    if use_intrabar and INTRABAR_PATH.exists():
        ib = pd.read_parquet(INTRABAR_PATH)
        ib = ib.rename(columns={"bar_ts": "ts_s"})
        n_before = len(merged)
        merged = merged.merge(
            ib[["ts_s", "p_low", "p_high"]],
            on="ts_s",
            how="left",
        )
        matched = merged["p_low"].notna().sum()
        merged.loc[merged["p_low"].notna(), "has_intrabar"] = True
        merged.loc[merged["p_low"].notna(), "intrabar_low"] = merged.loc[merged["p_low"].notna(), "p_low"]
        merged.loc[merged["p_high"].notna(), "intrabar_high"] = merged.loc[merged["p_high"].notna(), "p_high"]
        merged = merged.drop(columns=["p_low", "p_high"], errors="ignore")
        print(f"Intra-bar data: {matched}/{n_before} bars matched ({matched/n_before*100:.1f}%)")
    else:
        if use_intrabar:
            print(f"⚠️  Intra-bar 数据不存在: {INTRABAR_PATH}")
            print(f"   运行 python3 scripts/pull_polymarket_1m_intrabar.py 拉取")
        print(f"   使用 next-bar 快照模式 (精度较低)")

    return merged


def extract_signals(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """提取信号: eff_prob >= threshold, 计算 fill_price, intra-bar low, is_correct."""
    eff_prob = np.where(df["pred_up"] == 1, df["pred_prob"], 1.0 - df["pred_prob"])
    mask = eff_prob >= threshold
    sig = df[mask].copy()
    sig["eff_prob"] = eff_prob[mask]

    # 当前 bar 的 token 价格 (taker fill 判定用)
    sig["fill_price"] = np.where(
        sig["pred_up"] == 1,
        sig["raw_p_last"],
        1.0 - sig["raw_p_last"],
    )
    # Next-bar token price (fallback maker fill proxy)
    sig["fill_price_next"] = np.where(
        sig["pred_up"] == 1,
        sig["raw_p_last_next"],
        1.0 - sig["raw_p_last_next"],
    )
    # Intra-bar low: 用于精确 maker fill 判定
    # 对 UP 预测: low = intrabar_low (UP token 最低价)
    # 对 DOWN 预测: low = 1 - intrabar_high (DOWN token 最低价 = 1 - UP最高价)
    sig["maker_low"] = np.where(
        sig["pred_up"] == 1,
        sig["intrabar_low"],
        1.0 - sig["intrabar_high"],
    )
    sig["has_intrabar"] = sig["has_intrabar"].fillna(False)

    sig["is_correct"] = (sig["pred_up"] == sig["actual_up"]).astype(int)

    if COOLDOWN_BARS > 0:
        idxs = sig.index.values
        keep = np.ones(len(idxs), dtype=bool)
        last_kept = -COOLDOWN_BARS - 1
        for i, idx in enumerate(idxs):
            if idx - last_kept > COOLDOWN_BARS:
                last_kept = idx
            else:
                keep[i] = False
        sig = sig[keep]

    # Drop signals with missing next-bar price (fallback)
    sig = sig.dropna(subset=["fill_price_next"]).reset_index(drop=True)
    return sig


def _polymarket_taker_fee(p: np.ndarray) -> np.ndarray:
    """Polymarket taker fee rate: 0.25 × (p × (1-p))²"""
    return 0.25 * (p * (1.0 - p)) ** 2


def taker_metrics(signals: pd.DataFrame, limit_price: float, bet: float) -> dict:
    """TAKER: fill when fill_price <= limit_price (cross spread).
    Fee deducted from tokens (Polymarket 官方机制): tokens_after = tokens × (1 - feeRate)."""
    fp = signals["fill_price"].values
    ic = signals["is_correct"].values
    filled = fp <= limit_price
    n_sig = len(fp)
    n_fill = filled.sum()
    if n_fill == 0:
        return _empty("taker", limit_price, n_sig)

    fp_f = fp[filled]
    ic_f = ic[filled]
    fee_rates = _polymarket_taker_fee(fp_f)
    # 赢: tokens*(1-fee) - bet; 输: -bet (代币归零，费已含在减少的份额中)
    pnl = np.where(
        ic_f == 1,
        bet * ((1.0 - fee_rates) / fp_f - 1.0),
        -bet,
    )
    return {
        "scenario": "taker",
        "limit_price": limit_price,
        "n_signals": n_sig,
        "n_filled": int(n_fill),
        "fill_rate": n_fill / n_sig,
        "wins": int(ic_f.sum()),
        "losses": int(n_fill - ic_f.sum()),
        "win_rate": float(ic_f.mean()),
        "total_pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()),
        "cost_pct": float(fee_rates.mean()),
    }


def maker_metrics(signals: pd.DataFrame, limit_price: float, bet: float) -> dict:
    """
    MAKER (Post-only): 只考虑信号时价格 > limit 的情况（挂单不会立即成交）。
    成交判定:
      1. 有 intra-bar low → bar 内最低价 <= limit_price → 成交
      2. 无 intra-bar low → 用 next-bar 快照做近似（fallback）
    """
    fp = signals["fill_price"].values
    fp_next = signals["fill_price_next"].values
    maker_low = signals["maker_low"].values
    has_ib = signals["has_intrabar"].values.astype(bool)
    ic = signals["is_correct"].values

    # Post-only: 信号时价格 > limit → 挂单不会立即吃单
    post_only_eligible = fp > limit_price
    n_eligible = post_only_eligible.sum()
    n_sig = len(fp)

    # 成交判定: intra-bar low <= limit (精确) 或 next-bar <= limit (回退)
    fill_by_intrabar = post_only_eligible & has_ib & (maker_low <= limit_price)
    fill_by_fallback = post_only_eligible & ~has_ib & (fp_next <= limit_price)
    maker_fill = fill_by_intrabar | fill_by_fallback

    n_fill = int(maker_fill.sum())
    n_fill_ib = int(fill_by_intrabar.sum())
    n_fill_fb = int(fill_by_fallback.sum())

    if n_fill == 0:
        d = _empty("maker", limit_price, n_sig, n_eligible=int(n_eligible))
        d["n_fill_intrabar"] = 0
        d["n_fill_fallback"] = 0
        return d

    ic_f = ic[maker_fill]
    # Maker: 0 手续费，代币不扣费。返利单独计算（近似为 bet × rebate_rate）
    pnl = np.where(
        ic_f == 1,
        bet / limit_price - bet + bet * REBATE_MAKER_PCT,   # tokens×$1 - bet + rebate
        -bet + bet * REBATE_MAKER_PCT,                       # 代币归零 - bet + rebate
    )
    return {
        "scenario": "maker",
        "limit_price": limit_price,
        "n_signals": n_sig,
        "n_eligible": int(n_eligible),
        "n_filled": n_fill,
        "n_fill_intrabar": n_fill_ib,
        "n_fill_fallback": n_fill_fb,
        "fill_rate": n_fill / max(n_eligible, 1),
        "fill_rate_of_all": n_fill / n_sig,
        "wins": int(ic_f.sum()),
        "losses": int(n_fill - ic_f.sum()),
        "win_rate": float(ic_f.mean()),
        "total_pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()),
        "cost_pct": -REBATE_MAKER_PCT,
    }


def _empty(scenario: str, limit_price: float, n_sig: int, n_eligible: int = None) -> dict:
    d = {
        "scenario": scenario,
        "limit_price": limit_price,
        "n_signals": n_sig,
        "n_filled": 0,
        "fill_rate": 0.0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "total_pnl": 0.0,
        "avg_pnl": 0.0,
        "cost_pct": FEE_TAKER_PCT if scenario == "taker" else -REBATE_MAKER_PCT,
    }
    if n_eligible is not None:
        d["n_eligible"] = n_eligible
        d["fill_rate_of_all"] = 0.0
    return d


def run_comparison(signals: pd.DataFrame, bet: float) -> pd.DataFrame:
    """Run TAKER and MAKER for each limit price in range; return combined table."""
    prices = np.arange(LIMIT_LO, LIMIT_HI + LIMIT_STEP / 2, LIMIT_STEP)
    rows = []
    for p in prices:
        rows.append(taker_metrics(signals, float(p), bet))
        m = maker_metrics(signals, float(p), bet)
        if "n_eligible" in m:
            m["fill_rate"] = m["fill_rate_of_all"]  # for table: fill vs all signals
        rows.append(m)
    return pd.DataFrame(rows)


def print_table(comparison: pd.DataFrame, signals: pd.DataFrame, bet: float) -> None:
    """Print TAKER vs MAKER comparison table and viability summary."""
    w = 90
    n_ib = signals["has_intrabar"].sum()
    n_total = len(signals)

    # 数据来源
    data_mode = f"1分钟intra-bar" if n_ib > 0 else "next-bar快照(低精度)"
    ib_pct = n_ib / n_total * 100 if n_total > 0 else 0

    print()
    print("=" * w)
    print("  POST-ONLY MAKER vs TAKER — GRU_ETH_54 (prob≥0.52, limit 0.45–0.56)")
    print(f"  信号数: {n_total}  |  每单: ${bet:.0f}  |  Taker fee: {FEE_TAKER_PCT:.2%}  |  Maker rebate: {REBATE_MAKER_PCT:.2%}")
    print(f"  Maker成交判定: {data_mode} ({n_ib}/{n_total} bar有intra-bar数据, {ib_pct:.1f}%)")
    print("=" * w)

    taker = comparison[comparison["scenario"] == "taker"]
    maker = comparison[comparison["scenario"] == "maker"]

    print(f"\n  {'Limit':>6}  {'Mode':>6}  {'Fill%':>6}  {'Filled':>6}  {'(IB/FB)':>9}  {'WinRate':>7}  {'TotalPnL':>10}  {'AvgPnL':>8}")
    print("  " + "-" * 76)
    for lim in [0.46, 0.48, 0.50, 0.52, 0.53, 0.54, 0.55]:
        rt = taker[np.abs(taker["limit_price"] - lim) < 0.001]
        rm = maker[np.abs(maker["limit_price"] - lim) < 0.001]
        if len(rt):
            r = rt.iloc[0]
            print(f"  ${lim:.2f}   {'TAKER':>6}  {r['fill_rate']*100:5.1f}%  {int(r['n_filled']):6d}  {'':>9}  {r['win_rate']*100:6.1f}%  ${r['total_pnl']:+9.2f}  ${r['avg_pnl']:+7.2f}")
        if len(rm):
            r = rm.iloc[0]
            fill_pct = r.get("fill_rate_of_all", r["fill_rate"]) * 100
            ib_str = f"{int(r.get('n_fill_intrabar',0))}/{int(r.get('n_fill_fallback',0))}"
            print(f"  ${lim:.2f}   {'MAKER':>6}  {fill_pct:5.1f}%  {int(r['n_filled']):6d}  {ib_str:>9}  {r['win_rate']*100:6.1f}%  ${r['total_pnl']:+9.2f}  ${r['avg_pnl']:+7.2f}")
        if len(rt) or len(rm):
            print("  " + "-" * 76)

    # Best limit
    best_taker_idx = taker["total_pnl"].idxmax()
    best_maker_idx = maker["total_pnl"].idxmax()
    best_t = taker.loc[best_taker_idx]
    best_m = maker.loc[best_maker_idx]

    print("\n  " + "=" * w)
    print("  最佳限价 (按 PnL)")
    print("  " + "=" * w)
    print(f"  TAKER  limit=${best_t['limit_price']:.2f}:  fill={best_t['fill_rate']*100:.1f}%  wins={best_t['wins']:.0f}  PnL=${best_t['total_pnl']:+.2f}  avg=${best_t['avg_pnl']:+.2f}")
    maker_fill_pct = best_m.get("fill_rate_of_all", best_m["fill_rate"]) * 100
    print(f"  MAKER  limit=${best_m['limit_price']:.2f}:  fill={maker_fill_pct:.1f}%  wins={best_m['wins']:.0f}  PnL=${best_m['total_pnl']:+.2f}  avg=${best_m['avg_pnl']:+.2f}")

    # 对比: 如果相同限价下 maker vs taker
    lim053 = 0.53
    rt53 = taker[np.abs(taker["limit_price"] - lim053) < 0.001]
    rm53 = maker[np.abs(maker["limit_price"] - lim053) < 0.001]
    if len(rt53) and len(rm53):
        t53 = rt53.iloc[0]
        m53 = rm53.iloc[0]
        print(f"\n  同限价 ${lim053:.2f} 对比:")
        print(f"    TAKER: {int(t53['n_filled'])} 笔成交, PnL=${t53['total_pnl']:+.2f}, 胜率={t53['win_rate']*100:.1f}%")
        print(f"    MAKER: {int(m53['n_filled'])} 笔成交, PnL=${m53['total_pnl']:+.2f}, 胜率={m53['win_rate']*100:.1f}%")
        if int(m53["n_filled"]) > 0 and int(t53["n_filled"]) > 0:
            ratio = m53["total_pnl"] / t53["total_pnl"] if t53["total_pnl"] != 0 else float("inf")
            print(f"    Maker PnL / Taker PnL = {ratio:.2f}x")

    print()
    print("  " + "-" * w)
    print("  POST-ONLY MAKER 可行性判断")
    print("  " + "-" * w)

    avg_fill_maker = (maker["fill_rate_of_all"].mean() * 100) if "fill_rate_of_all" in maker.columns else (maker["fill_rate"].mean() * 100)
    avg_fill_taker = taker["fill_rate"].mean() * 100

    print(f"  • Taker 平均成交率: {avg_fill_taker:.1f}% (立即吃单)")
    print(f"  • Maker 平均成交率: {avg_fill_maker:.1f}% (等价格回落)")
    print(f"  • Taker 最佳 PnL:   ${best_t['total_pnl']:+.2f} (费用 ~{FEE_TAKER_PCT:.2%})")
    print(f"  • Maker 最佳 PnL:   ${best_m['total_pnl']:+.2f} (返利 ~{REBATE_MAKER_PCT:.2%})")

    if n_ib > 0:
        print(f"  • 数据精度: 使用 {n_ib} 个 bar 的 1 分钟 intra-bar low (高精度)")
    else:
        print(f"  • 数据精度: 仅用 next-bar 快照 (低精度，严重低估 maker 成交率)")

    # 结论判定
    if best_m["total_pnl"] > best_t["total_pnl"] * 0.8 and best_m.get("n_filled", 0) >= 30:
        print("  → 结论: Post-only Maker 有竞争力！成交率虽低但单笔利润高，总 PnL 接近或超过 Taker。")
        print("    建议: 实盘可考虑 Post-only 模式，省手续费 + 赚返利。")
    elif best_m.get("n_filled", 0) >= 30 and best_m["total_pnl"] > 0:
        ratio_str = f"{best_m['total_pnl']/best_t['total_pnl']:.1%}" if best_t["total_pnl"] > 0 else "N/A"
        print(f"  → 结论: Maker 可盈利但 PnL 仅为 Taker 的 {ratio_str}。")
        print("    建议: 混合模式 - 低限价用 Taker 确保成交，高限价用 Maker 省手续费。")
    elif best_m.get("n_filled", 0) < 30:
        print("  → 结论: Maker 成交次数太少，统计无意义。")
        if n_ib == 0:
            print("    原因: 无 intra-bar 数据，实际成交率应更高。请拉取 1 分钟数据后重试。")
        else:
            print("    原因: 即使有 intra-bar 数据，价格仍很少回落到限价。Taker 更适合。")
    else:
        print("  → 结论: Maker 亏损，不建议使用 Post-only 模式。")
    print("=" * w)


def main():
    ap = argparse.ArgumentParser(description="Post-only maker vs taker viability (GRU_ETH_54)")
    ap.add_argument("--no-cache", action="store_true", help="Regenerate GRU predictions and merge")
    ap.add_argument("--no-intrabar", action="store_true", help="强制不使用 intra-bar 数据 (用旧的 next-bar 快照)")
    args = ap.parse_args()

    use_ib = not args.no_intrabar
    merged = load_predictions_and_prices(use_cache=not args.no_cache, use_intrabar=use_ib)
    signals = extract_signals(merged, PROB_THRESHOLD)
    print(f"Signals (eff_prob >= {PROB_THRESHOLD}): {len(signals)}")

    comparison = run_comparison(signals, BET)
    print_table(comparison, signals, BET)


if __name__ == "__main__":
    main()
