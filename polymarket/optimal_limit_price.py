#!/usr/bin/env python3
"""
GRU 模型最优限价上限搜索（考虑逆向选择）
────────────────────────────────────────
方法：用模型自身的 prediction_trades.json 实际交易数据，
     按 tokenPrice 分组计算真实胜率和真实 PnL，
     扫描候选 LIMIT_PRICE，找到使累计 PnL 最大的价位。

与 optimize_buy_price.py 的区别：
  旧脚本：Exp 成交率曲线 × 整体 WR（假设 WR 不随价格变）
  本脚本：GRU 模型自身数据 → 每个价格的真实 WR → 自然包含逆向选择

用法：
  python3 polymarket/optimal_limit_price.py
"""

import json
import math
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent

MODELS = [
    {"dir": "logs_gru_eth_ek",          "name": "GRU_ETH_EK",          "init": 400},
    {"dir": "logs_gru_eth_no1h4h_ek",   "name": "GRU_ETH_no1h4h_EK",  "init": 400},
    {"dir": "logs_gru_btc_no1h4h_ek",   "name": "GRU_BTC_no1h4h_EK",  "init": 400},
    {"dir": "logs_gru_sol_ek",          "name": "GRU_SOL_EK",          "init": 400},
]

# 扫描范围
SCAN_LO = 0.40
SCAN_HI = 0.56
STEP = 0.01

# 图示条长度
BAR_MAX = 30

W = 72  # 输出宽度


def load_trades(log_dir: str) -> list:
    path = SCRIPT_DIR / log_dir / "prediction_trades.json"
    if not path.exists():
        return []
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return []
    out = []
    for t in data:
        if t.get("status") != "executed":
            continue
        try:
            price = round(float(t["tokenPrice"]), 2)
        except (TypeError, ValueError, KeyError):
            continue
        result = t.get("result")
        try:
            pnl = float(t["pnl"])
        except (TypeError, ValueError, KeyError):
            pnl = 0
        try:
            amount = float(t.get("amount") or 0)
        except (TypeError, ValueError):
            amount = 0
        try:
            conf = float(t.get("confidence") or 0)
        except (TypeError, ValueError):
            conf = 0
        out.append({
            "price": price,
            "result": result,
            "pnl": pnl,
            "amount": amount,
            "conf": conf,
        })
    return out


def make_bar(val, max_abs, positive=True):
    """生成文本条形图"""
    if max_abs == 0:
        return ""
    length = int(abs(val) / max_abs * BAR_MAX)
    length = max(length, 0)
    if val > 0:
        return "█" * length
    elif val < 0:
        return "░" * length
    return ""


def analyze_model(model: dict):
    name = model["name"]
    init = model["init"]
    trades = load_trades(model["dir"])

    if not trades:
        print(f"\n  ⚠ {name}: 无交易数据")
        return

    # ── 板块 1: 每个价格 bin 的边际统计 ──
    bins = defaultdict(lambda: {"w": 0, "l": 0, "pnl": 0.0, "n": 0,
                                "bets": [], "confs_w": [], "confs_l": []})
    for t in trades:
        b = bins[t["price"]]
        b["n"] += 1
        if t["result"] == "win":
            b["w"] += 1
            b["confs_w"].append(t["conf"])
        elif t["result"] == "lose":
            b["l"] += 1
            b["confs_l"].append(t["conf"])
        b["pnl"] += t["pnl"]
        b["bets"].append(t["amount"])

    print()
    print("=" * W)
    print(f"  {name} — 最优限价上限搜索（考虑逆向选择）")
    print(f"  数据: {len(trades)} 笔交易, 初始 ${init}")
    print("=" * W)

    # ── 板块 2: 边际分析（每个价格自身的 WR/PnL）──
    print()
    print("─" * W)
    print("  板块 1: 边际分析（各价格 bin 自身统计）")
    print("─" * W)
    print(f"  {'价格':>5s}  {'笔数':>4s}  {'胜':>3s}  {'负':>3s}"
          f"  {'胜率':>6s}  {'PnL':>8s}  {'均bet':>6s}"
          f"  {'Edge':>6s}")
    print(f"  {'─'*5}  {'─'*4}  {'─'*3}  {'─'*3}"
          f"  {'─'*6}  {'─'*8}  {'─'*6}"
          f"  {'─'*6}")

    prices_sorted = sorted(bins.keys())
    for p in prices_sorted:
        if p < SCAN_LO or p > SCAN_HI:
            continue
        b = bins[p]
        total = b["w"] + b["l"]
        wr = b["w"] / total * 100 if total > 0 else 0
        avg_bet = sum(b["bets"]) / len(b["bets"]) if b["bets"] else 0
        # Edge = WR × (1/p - 1) - (1 - WR)
        if total > 0:
            wr_f = b["w"] / total
            edge = wr_f * (1 / p - 1) - (1 - wr_f)
        else:
            edge = 0
        print(f"  {p:>5.2f}  {b['n']:>4d}  {b['w']:>3d}  {b['l']:>3d}"
              f"  {wr:>5.1f}%  {b['pnl']:>+8.1f}  {avg_bet:>6.1f}"
              f"  {edge:>+.3f}")

    # ── 板块 3: 累计扫描 ──
    #   如果 LIMIT_PRICE = X，则包含所有 tokenPrice ≤ X 的交易
    print()
    print("─" * W)
    print("  板块 2: 累计扫描（LIMIT_PRICE=X → 包含 ≤X 全部交易）")
    print("─" * W)
    print(f"  {'LIMIT':>5s}  {'笔数':>4s}  {'胜':>3s}  {'负':>3s}"
          f"  {'胜率':>6s}  {'累计PnL':>9s}  {'PnL/笔':>7s}"
          f"  图示")
    print(f"  {'─'*5}  {'─'*4}  {'─'*3}  {'─'*3}"
          f"  {'─'*6}  {'─'*9}  {'─'*7}"
          f"  {'─'*BAR_MAX}")

    scan_results = []
    cum_w = cum_l = cum_n = 0
    cum_pnl = 0.0

    price_candidates = []
    p = SCAN_LO
    while p <= SCAN_HI + 0.001:
        price_candidates.append(round(p, 2))
        p += STEP

    for limit_p in price_candidates:
        # 加入这个价格 bin 的数据
        if limit_p in bins:
            b = bins[limit_p]
            cum_w += b["w"]
            cum_l += b["l"]
            cum_n += b["n"]
            cum_pnl += b["pnl"]

        total = cum_w + cum_l
        wr = cum_w / total * 100 if total > 0 else 0
        ppn = cum_pnl / cum_n if cum_n > 0 else 0

        scan_results.append({
            "limit": limit_p,
            "n": cum_n, "w": cum_w, "l": cum_l,
            "wr": wr, "pnl": cum_pnl, "ppn": ppn,
        })

    # 找最大 PnL
    max_pnl = max(r["pnl"] for r in scan_results) if scan_results else 0
    min_pnl = min(r["pnl"] for r in scan_results) if scan_results else 0
    max_abs = max(abs(max_pnl), abs(min_pnl), 0.01)
    best = max(scan_results, key=lambda r: r["pnl"]) if scan_results else None

    for r in scan_results:
        if r["n"] == 0:
            continue
        bar = make_bar(r["pnl"], max_abs)
        marker = ""
        if best and r["limit"] == best["limit"]:
            marker = " ◄── 最优"
        elif r["limit"] == 0.46:
            marker = " (当前)"
        elif r["limit"] == 0.48:
            marker = " (旧)"

        print(f"  {r['limit']:>5.2f}  {r['n']:>4d}  {r['w']:>3d}"
              f"  {r['l']:>3d}  {r['wr']:>5.1f}%"
              f"  {r['pnl']:>+9.1f}  {r['ppn']:>+7.2f}"
              f"  {bar}{marker}")

    # ── 板块 4: 结论 ──
    print()
    print("─" * W)
    print("  板块 3: 结论")
    print("─" * W)

    if best:
        # 当前 0.48 的数据
        cur = next((r for r in scan_results if r["limit"] == 0.46), None)
        old = next((r for r in scan_results if r["limit"] == 0.48), None)

        print(f"  最优 LIMIT_PRICE: ${best['limit']:.2f}")
        print(f"    笔数 {best['n']}, 胜率 {best['wr']:.1f}%,"
              f" 累计PnL ${best['pnl']:+.1f},"
              f" PnL/笔 ${best['ppn']:+.2f}")
        print()

        if cur:
            print(f"  当前 $0.46:")
            print(f"    笔数 {cur['n']}, 胜率 {cur['wr']:.1f}%,"
                  f" 累计PnL ${cur['pnl']:+.1f},"
                  f" PnL/笔 ${cur['ppn']:+.2f}")
        if old:
            print(f"  旧设定 $0.48:")
            print(f"    笔数 {old['n']}, 胜率 {old['wr']:.1f}%,"
                  f" 累计PnL ${old['pnl']:+.1f},"
                  f" PnL/笔 ${old['ppn']:+.2f}")

        if cur and best["limit"] != 0.46:
            if best["pnl"] != 0 and cur["pnl"] != 0:
                improv = (best["pnl"] - cur["pnl"]) / abs(cur["pnl"])
                print()
                print(f"  改为 ${best['limit']:.2f} vs 当前 $0.46:"
                      f" PnL {improv:+.0%}")
            elif cur["pnl"] == 0:
                print()
                print(f"  当前 $0.46 PnL≈0, 改为 ${best['limit']:.2f}"
                      f" 可获 ${best['pnl']:+.1f}")

    # ── 板块 5: 逆向选择可视化 ──
    print()
    print("─" * W)
    print("  板块 4: 逆向选择可视化（胜率 vs 价格）")
    print("─" * W)

    # 使用 3-bin 滚动窗口平滑
    smoothed = []
    all_prices = sorted(bins.keys())
    relevant = [p for p in all_prices if SCAN_LO <= p <= SCAN_HI]

    for i, p in enumerate(relevant):
        # 取 [i-1, i, i+1] 的合并数据
        window_w = 0
        window_l = 0
        for j in range(max(0, i - 1), min(len(relevant), i + 2)):
            wp = relevant[j]
            window_w += bins[wp]["w"]
            window_l += bins[wp]["l"]
        window_total = window_w + window_l
        wr = window_w / window_total * 100 if window_total > 0 else 0
        smoothed.append((p, wr, window_total))

    print(f"  {'价格':>5s}  {'平滑WR':>7s}  {'样本':>4s}  图示")
    print(f"  {'─'*5}  {'─'*7}  {'─'*4}  {'─'*40}")

    for p, wr, n in smoothed:
        bar_len = int(wr / 100 * 40)
        bar = "█" * bar_len
        # 50% 参考线
        ref = "│" if bar_len < 20 else ""
        marker = ""
        if p == 0.46:
            marker = " ◄ 当前"
        elif p == 0.48:
            marker = " ◄ 旧"
        print(f"  {p:>5.2f}  {wr:>6.1f}%  {n:>4d}  {bar}{marker}")

    # 50% 参考线
    print(f"  {'':>5s}  {'':>7s}  {'':>4s}  "
          f"{'·'*20}50%{'·'*17}")

    # 计算相关性
    if len(smoothed) >= 3:
        xs = [p for p, wr, n in smoothed if n >= 3]
        ys = [wr for p, wr, n in smoothed if n >= 3]
        if len(xs) >= 3:
            mean_x = sum(xs) / len(xs)
            mean_y = sum(ys) / len(ys)
            cov = sum((x - mean_x) * (y - mean_y)
                      for x, y in zip(xs, ys)) / len(xs)
            std_x = (sum((x - mean_x)**2 for x in xs) / len(xs)) ** 0.5
            std_y = (sum((y - mean_y)**2 for y in ys) / len(ys)) ** 0.5
            if std_x > 0 and std_y > 0:
                corr = cov / (std_x * std_y)
                judge = ("✓ 正相关: 价格越高胜率越高 → 逆向选择"
                         if corr > 0.2
                         else "— 弱/无相关: 逆向选择不显著"
                         if corr > -0.2
                         else "✓ 负相关: 价格越低胜率越高 → 无逆向选择")
                print(f"\n  价格-胜率相关系数: {corr:+.3f}  {judge}")

    print()


def main():
    for m in MODELS:
        analyze_model(m)
    print("=" * W)
    print("  分析完成")
    print("=" * W)


if __name__ == "__main__":
    main()
