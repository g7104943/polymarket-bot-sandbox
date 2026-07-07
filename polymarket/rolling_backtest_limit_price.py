#!/usr/bin/env python3
"""
滚动回测 — GRU EK 组合 (ETH/BTC/SOL @ 0.46) 最优限价搜索
═══════════════════════════════════════════════════════════
数据：
  ① GRU 模型预测 → pred_prob, actual_up  (via get_backtest_df_one_asset)
  ② Polymarket 真实代币价格 → raw_p_last   (polymarket_prob_eth_usdt.parquet)

方法：
  1) 128 天 ≈ 12K 条 15m bar，生成 GRU 预测并合并代币价格
  2) 候选限价 [0.42, 0.56] 步长 0.005 → 成交率 / 胜率 / PnL 全景
  3) 滚动窗口 (30d train / 15d test) 验证最优限价的稳定性
  4) Bootstrap 95% 置信区间

用法:
  python3 polymarket/rolling_backtest_limit_price.py           # 正常运行
  python3 polymarket/rolling_backtest_limit_price.py --no-cache # 强制重新生成预测
"""

import sys
import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════
DATA_SRC = PROJECT_ROOT / "data" / "raw"
POLY_PROB_PATH = PROJECT_ROOT / "data" / "sentiment" / "polymarket_prob_eth_usdt.parquet"
CACHE_PATH = PROJECT_ROOT / "data" / "_cache_rolling_bt_eth.parquet"

POLY_START = "2025-10-08"   # Polymarket 15m 市场约 2025-10-09 开始
POLY_END   = "2026-02-16"   # 数据末尾（含）

# GRU EK 组合的交易参数（与 trader_configs.json 一致）
MODELS = [
    {"name": "GRU_ETH_EK",          "threshold": 0.54, "capital": 400, "bet_pct": 0.05},
    {"name": "GRU_ETH_no1h4h_EK",   "threshold": 0.54, "capital": 400, "bet_pct": 0.05},
]

# 限价扫描
SCAN_LO   = 0.42
SCAN_HI   = 0.60
SCAN_STEP = 0.005

# 滚动窗口
TRAIN_DAYS = 30
TEST_DAYS  = 15
SLIDE_DAYS = 15

# 交易成本
FEE_RATE  = 0.002     # Polymarket 手续费
SLIPPAGE  = 0.003     # 滑点
TOTAL_COST = FEE_RATE + SLIPPAGE

# Cooldown
COOLDOWN_BARS = 1

# Bootstrap
N_BOOTSTRAP = 2000

# 输出宽度
W = 82

# ═══════════════════════════════════════════════════════════
# Phase 1: 数据加载与合并
# ═══════════════════════════════════════════════════════════

def load_predictions_and_prices(use_cache: bool = True) -> pd.DataFrame:
    """
    生成 GRU ETH 预测 → 合并 Polymarket raw_p_last (代币价格)。
    带缓存：首次运行需 GRU 推理 (~20s)，之后秒级加载。
    """
    if use_cache and CACHE_PATH.exists():
        df = pd.read_parquet(CACHE_PATH)
        print(f"✅ 从缓存加载: {len(df)} 条  ({CACHE_PATH.name})")
        return df

    import torch
    from scripts.backtest_gru_regime import get_backtest_df_one_asset

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"⏳ 生成 GRU ETH 预测 (device={device}) ...")
    t0 = time.time()
    pred_df = get_backtest_df_one_asset(
        asset="ETH_USDT",
        test_days=365,
        device=device,
        data_src=DATA_SRC,
        after_date=POLY_START,
        end_date=POLY_END,
    )
    print(f"  → GRU 预测: {len(pred_df)} 条  ({time.time()-t0:.1f}s)")

    # timestamp (ms) → ts_s (秒) 用于合并
    if "timestamp" in pred_df.columns:
        pred_df["ts_s"] = (pred_df["timestamp"] // 1000).astype(np.int64)
    elif "date" in pred_df.columns:
        pred_df["ts_s"] = (
            pd.to_datetime(pred_df["date"]).astype("int64") // 10**9
        ).astype(np.int64)

    # 加载 Polymarket 代币价格
    poly = pd.read_parquet(POLY_PROB_PATH)
    poly["timestamp_s"] = poly["timestamp_s"].astype(np.int64)

    merged = pred_df.merge(
        poly[["timestamp_s", "raw_p_last"]],
        left_on="ts_s",
        right_on="timestamp_s",
        how="inner",
    )
    # 确保有 date 列 (datetime)
    if "date" not in merged.columns:
        merged["date"] = pd.to_datetime(merged["ts_s"], unit="s", utc=True)
    else:
        merged["date"] = pd.to_datetime(merged["date"], utc=True)

    merged = merged.sort_values("ts_s").reset_index(drop=True)
    print(f"  → 合并: {len(merged)} 条 (有 Polymarket 价格)")

    # 缓存
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cols_keep = ["ts_s", "date", "pred_prob", "pred_up", "actual_up", "raw_p_last"]
    cols_keep = [c for c in cols_keep if c in merged.columns]
    merged[cols_keep].to_parquet(CACHE_PATH, index=False)
    print(f"  → 缓存已保存: {CACHE_PATH.name}")
    return merged[cols_keep]


# ═══════════════════════════════════════════════════════════
# Phase 2: 信号提取 + 向量化 PnL 引擎
# ═══════════════════════════════════════════════════════════

def extract_signals(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    提取满足置信阈值的信号，计算:
      - fill_price: 买入侧的代币价格 (UP → raw_p_last, DOWN → 1-raw_p_last)
      - is_correct: 预测是否正确
    """
    eff_prob = np.where(df["pred_up"] == 1, df["pred_prob"], 1.0 - df["pred_prob"])
    mask = eff_prob >= threshold
    sig = df[mask].copy()
    sig["eff_prob"] = eff_prob[mask]

    # 买 YES (pred_up=1) 时代币价格 = raw_p_last
    # 买 NO  (pred_up=0) 时代币价格 ≈ 1 - raw_p_last
    sig["fill_price"] = np.where(
        sig["pred_up"] == 1,
        sig["raw_p_last"],
        1.0 - sig["raw_p_last"],
    )
    sig["is_correct"] = (sig["pred_up"] == sig["actual_up"]).astype(int)

    # Cooldown: 至少间隔 COOLDOWN_BARS
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

    return sig.reset_index(drop=True)


def vectorized_pnl(signals: pd.DataFrame, limit_price: float, bet: float) -> dict:
    """
    向量化计算某限价下的成交情况和总 PnL（固定下注额模式）。

    PnL 规则 (Polymarket):
      赢: +bet*(1/fill_price - 1 - TOTAL_COST)（手续费在买入时从份额扣除）
      输: -bet（不另扣费）
    """
    fp = signals["fill_price"].values
    ic = signals["is_correct"].values

    filled_mask = fp <= limit_price
    n_signals = len(fp)
    n_filled = filled_mask.sum()

    if n_filled == 0:
        return _empty_result(limit_price, n_signals)

    fp_filled = fp[filled_mask]
    ic_filled = ic[filled_mask]

    pnl_per = np.where(
        ic_filled == 1,
        bet * (1.0 / fp_filled - 1.0 - TOTAL_COST),
        -bet,  # Polymarket: 输不另扣费
    )
    wins = int(ic_filled.sum())
    losses = n_filled - wins
    total_pnl = float(pnl_per.sum())
    avg_fill_p = float(fp_filled.mean())

    # 成交信号 vs 未成交信号的平均置信
    ep = signals["eff_prob"].values
    avg_conf_fill = float(ep[filled_mask].mean())
    avg_conf_miss = float(ep[~filled_mask].mean()) if (~filled_mask).sum() > 0 else np.nan

    return {
        "limit_price": limit_price,
        "n_signals": n_signals,
        "n_filled": n_filled,
        "fill_rate": n_filled / n_signals,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / n_filled,
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / n_filled,
        "avg_fill_price": avg_fill_p,
        "avg_conf_fill": avg_conf_fill,
        "avg_conf_miss": avg_conf_miss,
    }


def _empty_result(lp, n_sig):
    return {
        "limit_price": lp, "n_signals": n_sig, "n_filled": 0,
        "fill_rate": 0, "wins": 0, "losses": 0, "win_rate": 0,
        "total_pnl": 0, "avg_pnl": 0, "avg_fill_price": np.nan,
        "avg_conf_fill": np.nan, "avg_conf_miss": np.nan,
    }


def scan_limit_prices(signals: pd.DataFrame, bet: float) -> pd.DataFrame:
    """在 [SCAN_LO, SCAN_HI] 扫描所有候选限价。"""
    prices = np.arange(SCAN_LO, SCAN_HI + SCAN_STEP / 2, SCAN_STEP)
    rows = [vectorized_pnl(signals, p, bet) for p in prices]
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# Phase 3: 滚动窗口验证
# ═══════════════════════════════════════════════════════════

def rolling_validation(signals: pd.DataFrame, bet: float) -> pd.DataFrame:
    """滚动窗口: train 找最优 P* → test 验证 OOS 表现。"""
    signals = signals.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(signals["date"])
    min_d, max_d = dates.min(), dates.max()

    results = []
    train_start = min_d
    while True:
        train_end = train_start + pd.Timedelta(days=TRAIN_DAYS)
        test_end  = train_end  + pd.Timedelta(days=TEST_DAYS)
        if test_end > max_d + pd.Timedelta(days=1):
            break

        train_mask = (dates >= train_start) & (dates < train_end)
        test_mask  = (dates >= train_end)   & (dates < test_end)
        train_sig = signals[train_mask]
        test_sig  = signals[test_mask]

        if len(train_sig) < 20 or len(test_sig) < 5:
            train_start += pd.Timedelta(days=SLIDE_DAYS)
            continue

        # Train: 找使 PnL 最大的 P*
        train_scan = scan_limit_prices(train_sig, bet)
        best_idx = train_scan["total_pnl"].idxmax()
        best_p   = train_scan.loc[best_idx, "limit_price"]
        train_r  = train_scan.loc[best_idx]

        # Test: 用 P* 评估
        test_r = vectorized_pnl(test_sig, best_p, bet)

        results.append({
            "train_start":  train_start.strftime("%m-%d"),
            "train_end":    train_end.strftime("%m-%d"),
            "test_end":     test_end.strftime("%m-%d"),
            "P*":           best_p,
            "tr_pnl":       train_r["total_pnl"],
            "tr_wr":        train_r["win_rate"],
            "tr_n":         train_r["n_filled"],
            "te_pnl":       test_r["total_pnl"],
            "te_wr":        test_r["win_rate"],
            "te_n":         test_r["n_filled"],
            "te_fill%":     test_r["fill_rate"],
        })
        train_start += pd.Timedelta(days=SLIDE_DAYS)

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════
# Phase 4: Bootstrap 置信区间
# ═══════════════════════════════════════════════════════════

def bootstrap_optimal_price(signals: pd.DataFrame, bet: float) -> dict:
    """Bootstrap 重采样 → 最优限价的分布和 95% CI。"""
    prices = np.arange(SCAN_LO, SCAN_HI + SCAN_STEP / 2, SCAN_STEP)
    n = len(signals)
    rng = np.random.RandomState(42)

    fp = signals["fill_price"].values
    ic = signals["is_correct"].values

    best_prices = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, size=n)
        fp_b = fp[idx]
        ic_b = ic[idx]
        best_pnl = -1e18
        best_p = 0.50
        for p in prices:
            mask = fp_b <= p
            if mask.sum() == 0:
                continue
            pnl = np.where(
                ic_b[mask] == 1,
                bet * (1.0 / fp_b[mask] - 1.0 - TOTAL_COST),
                -bet,
            ).sum()
            if pnl > best_pnl:
                best_pnl = pnl
                best_p = p
        best_prices.append(best_p)

    arr = np.array(best_prices)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "distribution": arr,
    }


# ═══════════════════════════════════════════════════════════
# Phase 5: 终端报告
# ═══════════════════════════════════════════════════════════

def _bar(val, max_val, width=20):
    if max_val <= 0 or val <= 0:
        return ""
    return "█" * max(0, int(val / max_val * width))


def print_report(name, scan_df, rolling_df, bs, signals, bet):
    """打印完整终端报告。"""
    print()
    print("═" * W)
    print(f"  {name}  —  滚动回测最优限价分析")
    print(f"  信号 {len(signals)} 个  |  128 天 Polymarket 真实代币价格  |  下注 ${bet:.0f}/笔")
    print("═" * W)

    # ── 1. 限价全景扫描 ──────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  ① 限价全景扫描  [{SCAN_LO:.2f} ~ {SCAN_HI:.2f}]  step={SCAN_STEP:.3f}")
    print(f"{'─'*W}")
    header = f"  {'限价':>6s} {'成交率':>6s} {'成交':>5s} {'胜率':>6s} {'PnL($)':>10s} {'均PnL':>7s}  图示"
    print(header)

    max_pnl = max(scan_df["total_pnl"].max(), 1)
    best_idx = scan_df["total_pnl"].idxmax()
    best_p = scan_df.loc[best_idx, "limit_price"]

    for i, row in scan_df.iterrows():
        mark = " ★" if abs(row["limit_price"] - best_p) < 0.001 else ""
        bar = _bar(row["total_pnl"], max_pnl)
        wr_str = f"{row['win_rate']*100:5.1f}%" if row["n_filled"] > 0 else "  — "
        avg_str = f"${row['avg_pnl']:+.2f}" if row["n_filled"] > 0 else "   —"
        print(
            f"  ${row['limit_price']:.3f}"
            f" {row['fill_rate']*100:5.1f}%"
            f" {int(row['n_filled']):5d}"
            f" {wr_str}"
            f" ${row['total_pnl']:+9.2f}"
            f" {avg_str:>7s}"
            f"  {bar}{mark}"
        )

    best_row = scan_df.loc[best_idx]
    print(
        f"\n  → 全局最优: ${best_p:.3f}"
        f"  PnL=${best_row['total_pnl']:+.2f}"
        f"  WR={best_row['win_rate']*100:.1f}%"
        f"  成交率={best_row['fill_rate']*100:.1f}%"
        f"  ({best_row['n_filled']:.0f}笔)"
    )

    # 与现行 0.48 对比
    row_046 = scan_df[np.abs(scan_df["limit_price"] - 0.46) < 0.001]
    if len(row_046) > 0:
        r46 = row_046.iloc[0]
        diff = best_row["total_pnl"] - r46["total_pnl"]
        print(
            f"  → 对比 $0.460 (现行):"
            f"  PnL=${r46['total_pnl']:+.2f}"
            f"  WR={r46['win_rate']*100:.1f}%"
            f"  差额=${diff:+.2f}"
        )

    # ── 2. 滚动窗口验证 ──────────────────────────────────
    if len(rolling_df) > 0:
        print(f"\n{'─'*W}")
        print(f"  ② 滚动窗口验证  train={TRAIN_DAYS}d / test={TEST_DAYS}d / slide={SLIDE_DAYS}d")
        print(f"{'─'*W}")
        h = (
            f"  {'窗口':>11s}  {'→test':>5s}"
            f"  {'P*':>6s}"
            f"  {'tr_PnL':>9s}  {'te_PnL':>9s}"
            f"  {'te_WR':>6s}  {'te成交':>5s}  {'te成交率':>6s}"
        )
        print(h)
        for _, rw in rolling_df.iterrows():
            te_wr = f"{rw['te_wr']*100:5.1f}%" if rw["te_n"] > 0 else "  —  "
            print(
                f"  {rw['train_start']}~{rw['train_end']}"
                f"  ~{rw['test_end']}"
                f"  ${rw['P*']:.3f}"
                f"  ${rw['tr_pnl']:+8.2f}"
                f"  ${rw['te_pnl']:+8.2f}"
                f"  {te_wr}"
                f"  {rw['te_n']:5.0f}"
                f"  {rw['te_fill%']*100:5.1f}%"
            )

        # 汇总
        avg_te_pnl = rolling_df["te_pnl"].mean()
        sum_te_pnl = rolling_df["te_pnl"].sum()
        med_p = rolling_df["P*"].median()
        std_p = rolling_df["P*"].std()
        pos_ratio = (rolling_df["te_pnl"] > 0).mean() * 100
        print(f"\n  → 测试窗口 PnL:  均值=${avg_te_pnl:+.2f}  合计=${sum_te_pnl:+.2f}  盈利窗口={pos_ratio:.0f}%")
        print(f"  → P* 分布:  中位=${med_p:.3f}  std=${std_p:.3f}  范围=[${rolling_df['P*'].min():.3f}, ${rolling_df['P*'].max():.3f}]")

    # ── 3. Bootstrap 置信区间 ─────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  ③ Bootstrap 95% CI  (n={N_BOOTSTRAP})")
    print(f"{'─'*W}")
    print(f"  最优限价:  均值=${bs['mean']:.3f}  中位=${bs['median']:.3f}  std=${bs['std']:.3f}")
    print(f"  95% CI:   [${bs['ci_lo']:.3f}, ${bs['ci_hi']:.3f}]")

    # 分布直方图 (简易文本版)
    dist = bs["distribution"]
    bins = np.arange(SCAN_LO, SCAN_HI + SCAN_STEP, SCAN_STEP)
    counts, edges = np.histogram(dist, bins=bins)
    max_c = max(counts.max(), 1)
    print(f"\n  分布直方图:")
    for i in range(len(counts)):
        mid = (edges[i] + edges[i + 1]) / 2
        bar_len = int(counts[i] / max_c * 30)
        if bar_len > 0 or abs(mid - bs["median"]) < SCAN_STEP:
            mark = " ◆" if abs(mid - bs["median"]) < SCAN_STEP else ""
            print(f"  ${mid:.3f} {'█' * bar_len} {counts[i]:>4d}{mark}")

    # ── 4. 逆向选择分析 ──────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  ④ 逆向选择分析")
    print(f"{'─'*W}")
    header_as = f"  {'限价':>6s}  {'成交':>5s}  {'成交WR':>6s}  {'未成交':>5s}  {'未成交WR':>6s}  {'WR差':>7s}  {'成交均置信':>8s}  {'未成交均置信':>8s}"
    print(header_as)

    for p in [0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54]:
        fill_mask = signals["fill_price"] <= p
        n_f = fill_mask.sum()
        n_u = (~fill_mask).sum()
        wr_f = signals.loc[fill_mask, "is_correct"].mean() if n_f > 0 else np.nan
        wr_u = signals.loc[~fill_mask, "is_correct"].mean() if n_u > 0 else np.nan
        diff = wr_f - wr_u if not (np.isnan(wr_f) or np.isnan(wr_u)) else np.nan
        cf = signals.loc[fill_mask, "eff_prob"].mean() if n_f > 0 else np.nan
        cu = signals.loc[~fill_mask, "eff_prob"].mean() if n_u > 0 else np.nan
        wr_f_s = f"{wr_f*100:5.1f}%" if not np.isnan(wr_f) else "   — "
        wr_u_s = f"{wr_u*100:5.1f}%" if not np.isnan(wr_u) else "   — "
        diff_s = f"{diff*100:+6.1f}%" if not np.isnan(diff) else "    — "
        cf_s   = f"{cf*100:5.1f}%  " if not np.isnan(cf) else "   —   "
        cu_s   = f"{cu*100:5.1f}%  " if not np.isnan(cu) else "   —   "
        mark = " ← 现行" if abs(p - 0.46) < 0.001 else ""
        print(f"  ${p:.2f}  {n_f:5d}  {wr_f_s}  {n_u:5d}  {wr_u_s}  {diff_s}  {cf_s}  {cu_s}{mark}")

    # ── 5. 综合建议 ──────────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  ⑤ 综合建议")
    print(f"{'─'*W}")

    # 推荐价 = bootstrap 中位数 (最鲁棒)
    rec_p = bs["median"]
    rec_scan = scan_df.iloc[(scan_df["limit_price"] - rec_p).abs().idxmin()]
    r46 = row_046.iloc[0] if len(row_046) > 0 else None

    print(f"  推荐限价: ${rec_p:.3f}")
    print(f"    PnL      = ${rec_scan['total_pnl']:+.2f}  (vs 现行 $0.46: ${r46['total_pnl']:+.2f})" if r46 is not None else f"    PnL      = ${rec_scan['total_pnl']:+.2f}")
    print(f"    胜率     = {rec_scan['win_rate']*100:.1f}%")
    print(f"    成交率   = {rec_scan['fill_rate']*100:.1f}%  ({rec_scan['n_filled']:.0f}/{rec_scan['n_signals']:.0f})")
    print(f"    95% CI   = [${bs['ci_lo']:.3f}, ${bs['ci_hi']:.3f}]")

    if r46 is not None:
        improve = rec_scan["total_pnl"] - r46["total_pnl"]
        print(f"    vs 0.46  = PnL +${improve:.2f}  ({improve/max(abs(r46['total_pnl']),0.01)*100:+.0f}%)")

    print("═" * W)


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="滚动回测 — GRU EK 组合最优限价搜索")
    ap.add_argument("--no-cache", action="store_true", help="强制重新生成 GRU 预测")
    args = ap.parse_args()

    print("=" * W)
    print("  滚动回测 — GRU EK 组合 (ETH @ 0.46) 最优限价搜索")
    print(f"  数据: Polymarket 代币价格 128 天 (2025-10 ~ 2026-02)")
    print(f"  扫描: [{SCAN_LO:.2f}, {SCAN_HI:.2f}] step={SCAN_STEP:.3f}")
    print(f"  滚动: train={TRAIN_DAYS}d / test={TEST_DAYS}d / slide={SLIDE_DAYS}d")
    print(f"  成本: fee={FEE_RATE:.1%} + slippage={SLIPPAGE:.1%} = {TOTAL_COST:.1%}")
    print("=" * W)

    # Phase 1: 加载数据
    merged = load_predictions_and_prices(use_cache=not args.no_cache)

    for cfg in MODELS:
        name = cfg["name"]
        threshold = cfg["threshold"]
        bet = cfg["capital"] * cfg["bet_pct"]  # $400 × 5% = $20

        # Phase 2: 提取信号
        signals = extract_signals(merged, threshold)
        print(f"\n  {name}: {len(signals)} 个信号 (threshold≥{threshold})")

        # Phase 2: 全景扫描
        scan_df = scan_limit_prices(signals, bet)

        # Phase 3: 滚动窗口
        rolling_df = rolling_validation(signals, bet)

        # Phase 4: Bootstrap
        print(f"  ⏳ Bootstrap ({N_BOOTSTRAP} 次) ...", end="", flush=True)
        t0 = time.time()
        bs = bootstrap_optimal_price(signals, bet)
        print(f" {time.time()-t0:.1f}s")

        # Phase 5: 报告
        print_report(name, scan_df, rolling_df, bs, signals, bet)


if __name__ == "__main__":
    main()
