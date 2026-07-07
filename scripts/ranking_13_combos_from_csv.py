#!/usr/bin/env python3
"""
从已生成的 hyperparam_tune_13_combos.csv 生成综合排序表。
- 只保留「稳健」行，约束：整段/冷验证/365天笔数满足最小值。
- 科学打分：综合得分 = 最终资金×(1−λ×回撤%/100)×(最低/初始%/100)，初始=400，按综合得分降序。
- 输出含 最低/初始%、365天_稳健 vs 365天_固定。

用法（项目根目录）:
  python scripts/ranking_13_combos_from_csv.py
  python scripts/ranking_13_combos_from_csv.py --csv path/to/hyperparam_tune_13_combos.csv --out path/to/out
"""
import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIN_TRADES_TOTAL = 30
MIN_TRADES_COLD = 10
RANK_LAMBDA_DD = 0.4  # 综合得分中回撤惩罚系数


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="hyperparam_tune_13_combos.csv 路径，默认项目根目录下")
    ap.add_argument("--out", default=None, help="输出目录，默认与 CSV 同目录")
    ap.add_argument("--min-trades-total", type=int, default=MIN_TRADES_TOTAL)
    ap.add_argument("--min-trades-cold", type=int, default=MIN_TRADES_COLD)
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else PROJECT_ROOT / "hyperparam_tune_13_combos.csv"
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path
    out_dir = Path(args.out) if args.out else csv_path.parent
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    robust = df.loc[df["组内排名"] == "稳健"].copy()
    if robust.empty:
        print("未找到 组内排名=稳健 的行")
        return

    mask = (robust["整段笔数"] >= args.min_trades_total) & (robust["冷验证笔数_稳健"] >= args.min_trades_cold)
    if "365天笔数_稳健" in robust.columns:
        mask = mask & (robust["365天笔数_稳健"] >= args.min_trades_total)
    rank_df = robust.loc[mask].copy()
    # 综合得分 = 最终资金 × (1 − λ×回撤%/100) × (最低/初始%/100)
    if "365天最终资金_稳健" in rank_df.columns and "365天最大回撤%_稳健" in rank_df.columns and "365天最低/初始%_稳健" in rank_df.columns:
        cap = rank_df["365天最终资金_稳健"].astype(float)
        dd = rank_df["365天最大回撤%_稳健"].astype(float)
        min_init = rank_df["365天最低/初始%_稳健"].astype(float)
    else:
        cap = rank_df["整段最终资金"].astype(float)
        dd = rank_df["整段最大回撤%"].astype(float)
        min_init = rank_df["整段最低/初始%"].astype(float) if "整段最低/初始%" in rank_df.columns else 100.0
    rank_df["综合得分"] = (cap * (1.0 - RANK_LAMBDA_DD * dd / 100.0) * (min_init / 100.0)).round(2)
    rank_df = rank_df.sort_values("综合得分", ascending=False).reset_index(drop=True)
    rank_df.insert(0, "综合排名", range(1, len(rank_df) + 1))
    if "365天最终资金_固定" in rank_df.columns:
        rank_df["未超参排名"] = rank_df["365天最终资金_固定"].rank(ascending=False, method="min").astype(int)
    out_cols = [
        "综合排名", "未超参排名", "组合", "综合得分", "阈值", "Δ", "p1", "p2", "p3", "p4", "校准",
        "365天最终资金_稳健", "365天胜率%_稳健", "365天最大回撤%_稳健", "365天最低/初始%_稳健", "365天笔数_稳健",
        "365天最终资金_固定", "365天胜率%_固定", "365天最大回撤%_固定", "365天最低/初始%_固定", "365天笔数_固定",
        "整段最终资金", "整段胜率%", "整段最大回撤%", "整段最低/初始%", "整段笔数",
        "冷验证最终资金_稳健", "冷验证胜率%_稳健", "冷验证回撤%_稳健", "冷验证最低/初始%_稳健", "冷验证笔数_稳健",
    ]
    rank_df = rank_df[[c for c in out_cols if c in rank_df.columns]]
    out_path = out_dir / "hyperparam_tune_13_combos_ranking.csv"
    rank_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("已写出: {} （{} 行，按 综合得分 排序，λ_dd={}）".format(out_path, len(rank_df), RANK_LAMBDA_DD))
    if len(robust) != len(rank_df):
        print("剔除 {} 个组合（未达最小笔数）".format(len(robust) - len(rank_df)))
    # 每个组合两行：_超参（365天_稳健）与 _未超参（365天_固定），按最终资金统一排名；列：排名, 币种, 周期, 交易次数, 胜场, 败场, 胜率%, 最终资金, 盈亏%, 最大回撤%, 最低/初始%, 阈值, 动态调整, 校准, 组合
    INITIAL_CAPITAL = 400.0
    def _combo_to_symbol(name):
        n = (name or "").lower()
        if "eth" in n: return "ETH"
        if "btc" in n: return "BTC"
        if "xrp" in n: return "XRP"
        if "sol" in n: return "SOL"
        return ""
    def _combo_has_dynamic(name):
        return "有" if name in ("logs_eth_10_90", "logs_gru_eth_54", "logs_gru_eth_55_dyn", "logs_gru_xrp_53_no1h4h") else "无"
    out_order = ["排名", "币种", "周期", "交易次数", "胜场", "败场", "胜率%", "最终资金", "盈亏%", "最大回撤%", "最低/初始%", "阈值", "动态调整", "校准", "组合"]
    rows = []
    for _, r in rank_df.iterrows():
        base = str(r.get("组合", ""))
        symbol = _combo_to_symbol(base)
        dyn = _combo_has_dynamic(base)
        # 超参行
        row_tuned = {"周期": "365天", "组合": base + "_超参", "校准": r.get("校准", "无"), "币种": symbol, "动态调整": dyn}
        if "阈值" in r.index:
            row_tuned["阈值"] = r["阈值"]
        for key, col in [
            ("交易次数", "365天笔数_稳健"),
            ("胜率%", "365天胜率%_稳健"),
            ("最终资金", "365天最终资金_稳健"),
            ("最大回撤%", "365天最大回撤%_稳健"),
            ("最低/初始%", "365天最低/初始%_稳健"),
        ]:
            if col in r.index and pd.notna(r.get(col)):
                row_tuned[key] = r[col]
        if "365天笔数_稳健" in r.index and "365天胜率%_稳健" in r.index:
            n, wr = float(r["365天笔数_稳健"]), float(r["365天胜率%_稳健"])
            row_tuned["胜场"] = int(round(n * wr / 100.0))
            row_tuned["败场"] = int(n) - row_tuned["胜场"]
        if "365天最终资金_稳健" in r.index:
            cap = float(r["365天最终资金_稳健"])
            row_tuned["盈亏%"] = round((cap - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0, 2)
        rows.append(row_tuned)
        # 未超参行
        row_fixed = {"周期": "365天", "组合": base + "_未超参", "校准": "无", "币种": symbol, "动态调整": dyn}
        if "阈值" in r.index:
            row_fixed["阈值"] = r["阈值"]
        for key, col in [
            ("交易次数", "365天笔数_固定"),
            ("胜率%", "365天胜率%_固定"),
            ("最终资金", "365天最终资金_固定"),
            ("最大回撤%", "365天最大回撤%_固定"),
            ("最低/初始%", "365天最低/初始%_固定"),
        ]:
            if col in r.index and pd.notna(r.get(col)):
                row_fixed[key] = r[col]
        if "365天笔数_固定" in r.index and "365天胜率%_固定" in r.index:
            n, wr = float(r["365天笔数_固定"]), float(r["365天胜率%_固定"])
            row_fixed["胜场"] = int(round(n * wr / 100.0))
            row_fixed["败场"] = int(n) - row_fixed["胜场"]
        if "365天最终资金_固定" in r.index:
            cap = float(r["365天最终资金_固定"])
            row_fixed["盈亏%"] = round((cap - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0, 2)
        rows.append(row_fixed)
    summary_df = pd.DataFrame(rows)
    if "最终资金" in summary_df.columns:
        summary_df = summary_df.sort_values("最终资金", ascending=False).reset_index(drop=True)
        summary_df.insert(0, "排名", range(1, len(summary_df) + 1))
    summary_df = summary_df[[c for c in out_order if c in summary_df.columns]]
    summary_path = out_dir / "超参排名一览.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("已写出: {} （{} 行，含超参/未超参，按最终资金统一排名）".format(summary_path, len(summary_df)))
    print("  回测下单价格: 0.527（Polymarket 15m 涨跌：赢=bet×(1/价−1)−fee，输=−bet−fee）")


if __name__ == "__main__":
    main()
