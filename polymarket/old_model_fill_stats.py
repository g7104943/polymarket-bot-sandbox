#!/usr/bin/env python3
"""
GRU 模型成交率 · 置信度 · 逆向选择分析（EK 组合 @ LIMIT=0.46）

板块：
  1. 基础统计（总交易、胜负、胜率、PnL、均token价）
  2. 按买入价分段统计（≤0.46 vs >0.46）
  3. 置信度分布（成交 vs 分段 WR）
  4. 价格监控成交率（从 trading_stdout.log 解析 累计统计）
  5. 与 Exp16_bp0460 / Exp13_bp0460 对比

用法：
  python3 polymarket/old_model_fill_stats.py
"""

import json
import re
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent

# ─── 配置 ───
COMBOS = [
    {
        "log_dir": "logs_gru_eth_ek",
        "name": "GRU_ETH_EK",
        "initial": 400,
        "limit_price": 0.46,
        "coin": "ETH",
    },
    {
        "log_dir": "logs_gru_eth_no1h4h_ek",
        "name": "GRU_ETH_no1h4h_EK",
        "initial": 400,
        "limit_price": 0.46,
        "coin": "ETH",
    },
    {
        "log_dir": "logs_gru_btc_no1h4h_ek",
        "name": "GRU_BTC_no1h4h_EK",
        "initial": 400,
        "limit_price": 0.46,
        "coin": "BTC",
    },
    {
        "log_dir": "logs_gru_sol_ek",
        "name": "GRU_SOL_EK",
        "initial": 400,
        "limit_price": 0.46,
        "coin": "SOL",
    },
]

# Exp 对比组（从 prediction_trades.json 读取）
EXP_COMPARE = [
    {
        "log_dir": "logs_v5_exp16_bp0460",
        "name": "Exp16_bp0460",
        "initial": 800,
        "coin_filter": None,
    },
    {
        "log_dir": "logs_v5_exp13_bp0460",
        "name": "Exp13_bp0460",
        "initial": 800,
        "coin_filter": None,
    },
]

PRICE_CUTOFF = 0.46  # 分段统计的价格分界线


# ─── 数据加载 ───
def load_trades(log_dir: str) -> list:
    path = SCRIPT_DIR / log_dir / "prediction_trades.json"
    if not path.exists():
        return []
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for t in data:
        if t.get("status") != "executed":
            continue
        symbol = (t.get("symbol") or "").upper().replace("/USDT", "")
        try:
            price = float(t["tokenPrice"])
        except (TypeError, ValueError, KeyError):
            price = None
        try:
            conf = float(t["confidence"])
        except (TypeError, ValueError, KeyError):
            conf = None
        try:
            pnl = float(t["pnl"])
        except (TypeError, ValueError, KeyError):
            pnl = None
        result = t.get("result")
        try:
            amount = float(t.get("amount") or 0)
        except (TypeError, ValueError):
            amount = 0
        slug = t.get("marketSlug", "")
        out.append({
            "symbol": symbol,
            "price": price,
            "conf": conf,
            "pnl": pnl,
            "result": result,
            "amount": amount,
            "slug": slug,
        })
    return out


def parse_log_fill_stats(log_dir: str) -> dict | None:
    """从 trading_stdout.log 解析最新 累计统计"""
    log_path = SCRIPT_DIR / log_dir / "trading_stdout.log"
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    lines = text.splitlines()
    pat = re.compile(
        r"累计统计:\s*成交率\s*([\d.]+)%\s*"
        r"\((\d+)/(\d+)\)\s*\|\s*"
        r"成交均置信\s*([\d.]+|-)%?\s*\|\s*"
        r"未成交均置信\s*([\d.]+|-)%?"
    )
    last = None
    for line in reversed(lines):
        m = pat.search(line)
        if m:
            last = m
            break
    if not last:
        return None
    fr, filled, total, cf, cu = last.groups()
    return {
        "fill_rate": float(fr),
        "filled": int(filled),
        "total": int(total),
        "conf_filled": float(cf) if cf != "-" else None,
        "conf_unfilled": float(cu) if cu != "-" else None,
    }


def parse_log_order_events(log_dir: str) -> dict:
    """统计挂单/部分成交/最终未买事件数"""
    log_path = SCRIPT_DIR / log_dir / "trading_stdout.log"
    counts = {"partial": 0, "full": 0, "missed": 0, "monitor": 0}
    if not log_path.exists():
        return counts
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return counts
    counts["partial"] = len(re.findall(r"挂单部分成交", text))
    counts["full"] = len(re.findall(r"全额成交", text))
    counts["missed"] = len(re.findall(r"最终未买", text))
    counts["monitor"] = len(re.findall(r"挂单中", text))
    return counts


# ─── 统计函数 ───
def compute_stats(trades: list, coin_filter: str = None) -> dict:
    """从 trades 列表计算统计"""
    if coin_filter:
        trades = [t for t in trades if t["symbol"] == coin_filter]
    if not trades:
        return None
    n = len(trades)
    wins = sum(1 for t in trades if t["result"] == "win")
    losses = sum(1 for t in trades if t["result"] == "lose")
    pending = n - wins - losses
    total_pnl = sum(t["pnl"] for t in trades if t["pnl"] is not None)
    prices = [t["price"] for t in trades if t["price"] is not None]
    avg_price = sum(prices) / len(prices) if prices else None
    confs = [t["conf"] for t in trades if t["conf"] is not None]
    avg_conf = sum(confs) / len(confs) if confs else None
    win_confs = [t["conf"] for t in trades
                 if t["result"] == "win" and t["conf"] is not None]
    lose_confs = [t["conf"] for t in trades
                  if t["result"] == "lose" and t["conf"] is not None]
    avg_win_conf = sum(win_confs) / len(win_confs) if win_confs else None
    avg_lose_conf = sum(lose_confs) / len(lose_confs) if lose_confs else None
    wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    amounts = [t["amount"] for t in trades if t["amount"]]
    avg_bet = sum(amounts) / len(amounts) if amounts else 0
    return {
        "n": n, "wins": wins, "losses": losses, "pending": pending,
        "wr": wr, "pnl": total_pnl, "avg_price": avg_price,
        "avg_conf": avg_conf, "avg_win_conf": avg_win_conf,
        "avg_lose_conf": avg_lose_conf, "avg_bet": avg_bet,
    }


def fmt_pct(v, w=6, as_prob=False):
    """格式化百分比。as_prob=True 时将 0.55 显示为 55.0%"""
    if v is None:
        return "—".rjust(w)
    return f"{v * 100:.1f}%" if as_prob else f"{v:.1f}%"


def fmt_dollar(v, w=8):
    return f"${v:+.2f}" if v is not None else "—".rjust(w)


def fmt_price(v, w=6):
    return f"${v:.4f}" if v is not None else "—".rjust(w)


# ─── 主输出 ───
def main():
    W = 72  # 输出宽度

    print()
    print("=" * W)
    print("  GRU 模型成交率·置信度·逆向选择分析")
    print("  (GRU EK 组合 @ LIMIT=0.46)")
    print("=" * W)

    for combo in COMBOS:
        log_dir = combo["log_dir"]
        name = combo["name"]
        initial = combo["initial"]
        coin = combo["coin"]

        trades = load_trades(log_dir)
        if not trades:
            print(f"\n  ⚠ {name}: 无交易数据")
            continue

        all_stats = compute_stats(trades)
        low = [t for t in trades
               if t["price"] is not None and t["price"] <= PRICE_CUTOFF]
        high = [t for t in trades
                if t["price"] is not None and t["price"] > PRICE_CUTOFF]
        low_stats = compute_stats(low) if low else None
        high_stats = compute_stats(high) if high else None

        log_fill = parse_log_fill_stats(log_dir)
        order_ev = parse_log_order_events(log_dir)

        # ─ 板块 1: 基础统计 ─
        print()
        print("─" * W)
        fc = initial + all_stats["pnl"]
        print(f"  ┌─ {name} ({coin}) ─ 初始${initial}"
              f" ─ 当前${fc:.0f}"
              f" ─ ROI {(fc - initial) / initial * 100:+.1f}%")
        print(f"  │")
        print(f"  │  总交易    {all_stats['n']:>4d} 笔"
              f"   胜/负  {all_stats['wins']}/{all_stats['losses']}"
              f"   胜率 {all_stats['wr']:.1f}%"
              f"   PnL {fmt_dollar(all_stats['pnl'])}")
        print(f"  │  均token价 {fmt_price(all_stats['avg_price'])}"
              f"   均bet  ${all_stats['avg_bet']:.1f}"
              f"   均置信 {fmt_pct(all_stats['avg_conf'], as_prob=True)}")
        print(f"  │")

        # ─ 板块 2: 按买入价分段 ─
        print(f"  │  ── 按买入价分段 ──")
        hdr = (f"  │  {'区间':<14s}"
               f"{'笔数':>5s}"
               f"{'胜/负':>8s}"
               f"{'胜率':>7s}"
               f"{'PnL':>10s}"
               f"{'均价':>8s}"
               f"{'均置信':>7s}")
        sep = f"  │  {'─' * 14}" + "─" * 45
        print(hdr)
        print(sep)
        for label, st in [
            (f"≤${PRICE_CUTOFF:.2f}", low_stats),
            (f" >${PRICE_CUTOFF:.2f}", high_stats),
        ]:
            if st is None:
                print(f"  │  {label:<14s}{'—':>5s}"
                      f"{'—':>8s}{'—':>7s}"
                      f"{'—':>10s}{'—':>8s}{'—':>7s}")
                continue
            wl = f"{st['wins']}/{st['losses']}"
            print(f"  │  {label:<14s}"
                  f"{st['n']:>5d}"
                  f"{wl:>8s}"
                  f"{st['wr']:>6.1f}%"
                  f"{fmt_dollar(st['pnl']):>10s}"
                  f"  {fmt_price(st['avg_price'])}"
                  f"{fmt_pct(st['avg_conf'], as_prob=True):>7s}")
        print(f"  │")

        # ─ 板块 3: 置信度分析 ─
        print(f"  │  ── 置信度分析 ──")
        print(f"  │  赢单均置信  "
              f"{fmt_pct(all_stats['avg_win_conf'], as_prob=True)}")
        print(f"  │  输单均置信  "
              f"{fmt_pct(all_stats['avg_lose_conf'], as_prob=True)}")
        if (all_stats["avg_win_conf"] is not None
                and all_stats["avg_lose_conf"] is not None):
            diff = (all_stats["avg_win_conf"]
                    - all_stats["avg_lose_conf"]) * 100
            judge = "✓ 正向" if diff > 0 else "⚠ 逆向"
            print(f"  │  差值        {diff:+.2f}%  {judge}"
                  f"（赢单置信 {'>' if diff > 0 else '<'} 输单置信）")
        print(f"  │")

        # ─ 板块 4: 价格监控成交率 ─
        print(f"  │  ── 价格监控（日志 累计统计）──")
        if log_fill:
            fr = f"{log_fill['fill_rate']:.1f}%"
            ratio = f"{log_fill['filled']}/{log_fill['total']}"
            cf = (f"{log_fill['conf_filled']:.1f}%"
                  if log_fill["conf_filled"] is not None else "—")
            cu = (f"{log_fill['conf_unfilled']:.1f}%"
                  if log_fill["conf_unfilled"] is not None else "—")
            if (log_fill["conf_filled"] is not None
                    and log_fill["conf_unfilled"] is not None):
                d = log_fill["conf_filled"] - log_fill["conf_unfilled"]
                ds = f"{d:+.1f}%"
            else:
                ds = "—"
            print(f"  │  成交率  {fr:>8s}"
                  f"  成交/总  {ratio:>8s}")
            print(f"  │  成交均置信  {cf:>7s}"
                  f"  未成交均置信  {cu:>7s}"
                  f"  差值  {ds:>7s}")
        else:
            print(f"  │  (无 累计统计 数据)")
        print(f"  │")

        # ─ 板块 5: 挂单事件 ─
        print(f"  │  ── 挂单事件统计 ──")
        print(f"  │  部分成交  {order_ev['partial']:>4d} 次"
              f"   全额成交  {order_ev['full']:>4d} 次")
        print(f"  │  挂单监控  {order_ev['monitor']:>4d} 次"
              f"   最终未买  {order_ev['missed']:>4d} 次")

        print(f"  └{'─' * (W - 3)}")

    # ━━━ Exp 对比 ━━━
    print()
    print("─" * W)
    print("  Exp 对比（bp0460 的 BTC 数据）")
    print("─" * W)
    hdr2 = (f"  {'模型':<20s}"
            f"{'BTC笔':>6s}"
            f"{'胜/负':>8s}"
            f"{'胜率':>7s}"
            f"{'BTC_PnL':>10s}"
            f"{'均价':>9s}"
            f"{'均置信':>7s}")
    sep2 = f"  {'─' * 20}" + "─" * 47
    print(hdr2)
    print(sep2)
    for exp in EXP_COMPARE:
        trades = load_trades(exp["log_dir"])
        btc_trades = [t for t in trades if t["symbol"] == "BTC"]
        st = compute_stats(btc_trades) if btc_trades else None
        if st is None:
            print(f"  {exp['name']:<20s}{'—':>6s}")
            continue
        wl = f"{st['wins']}/{st['losses']}"
        print(f"  {exp['name']:<20s}"
              f"{st['n']:>6d}"
              f"{wl:>8s}"
              f"{st['wr']:>6.1f}%"
              f"{fmt_dollar(st['pnl']):>10s}"
              f"  {fmt_price(st['avg_price'])}"
              f"{fmt_pct(st['avg_conf'], as_prob=True):>7s}")

    # 也显示 Exp16/13 bp0480 的全部币种和 ETH 数据
    print()
    print(f"  {'模型':<20s}"
          f"{'ETH笔':>6s}"
          f"{'胜/负':>8s}"
          f"{'胜率':>7s}"
          f"{'ETH_PnL':>10s}"
          f"{'均价':>9s}"
          f"{'均置信':>7s}")
    print(sep2)
    for exp in EXP_COMPARE:
        trades = load_trades(exp["log_dir"])
        eth_trades = [t for t in trades if t["symbol"] == "ETH"]
        st = compute_stats(eth_trades) if eth_trades else None
        if st is None:
            print(f"  {exp['name']:<20s}{'—':>6s}")
            continue
        wl = f"{st['wins']}/{st['losses']}"
        print(f"  {exp['name']:<20s}"
              f"{st['n']:>6d}"
              f"{wl:>8s}"
              f"{st['wr']:>6.1f}%"
              f"{fmt_dollar(st['pnl']):>10s}"
              f"  {fmt_price(st['avg_price'])}"
              f"{fmt_pct(st['avg_conf'], as_prob=True):>7s}")

    print()
    print("=" * W)
    print("  分析完成")
    print("=" * W)


if __name__ == "__main__":
    main()
