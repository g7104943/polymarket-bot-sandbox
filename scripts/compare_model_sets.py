"""
两套模型回测结果对比：并排看盈亏/胜率，可每日追加到 CSV 做长期对比。

用法：
  # 1) 先分别对两套模型跑回测并保存 JSON
  python scripts/backtest_simulation.py --models-dir data/models_setA --output-json reports/daily_a.json
  python scripts/backtest_simulation.py --models-dir data/models_setB --output-json reports/daily_b.json

  # 2) 对比两个 JSON
  python scripts/compare_model_sets.py --json-a reports/daily_a.json --json-b reports/daily_b.json

  # 3) 对比并追加到每日记录（用于长期看哪套更好）
  python scripts/compare_model_sets.py --json-a reports/daily_a.json --json-b reports/daily_b.json --append-csv reports/daily_comparison.csv

  # 4) 直接指定两套模型目录，自动跑回测再对比（会先执行两次 backtest_simulation）
  python scripts/compare_model_sets.py --models-dir-a data/models_setA --models-dir-b data/models_setB
  python scripts/compare_model_sets.py --models-dir-a data/models_setA --models-dir-b data/models_setB --append-csv reports/daily_comparison.csv --label-a setA --label-b setB
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _aggregate(data: dict) -> dict:
    results = data.get("results") or []
    total_trades = sum(r.get("total_trades", 0) for r in results)
    wins = sum(r.get("wins", 0) for r in results)
    losses = sum(r.get("losses", 0) for r in results)
    win_rate = (wins / total_trades * 100) if total_trades else 0.0
    pnls = [r["profit_pct"] for r in results if r.get("total_trades", 0) > 0]
    profit_pct = (sum(pnls) / len(pnls)) if pnls else 0.0
    max_dd = max((r.get("max_drawdown", 0) for r in results), default=0.0)
    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "profit_pct": profit_pct,
        "max_drawdown": max_dd,
    }


def _run_backtest(models_dir: str, output_json: str, output_html: str) -> dict:
    # 在项目根目录执行回测，--models-dir 若为相对路径则相对项目根
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "backtest_simulation.py"),
        "--models-dir", models_dir,
        "--output-json", str(output_json),
        "--output-html", str(output_html),
    ]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    return json.loads(Path(output_json).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description="两套模型回测结果对比，支持每日追加 CSV")
    ap.add_argument("--json-a", type=str, help="回测结果 JSON A（与 --models-dir-a 二选一）")
    ap.add_argument("--json-b", type=str, help="回测结果 JSON B（与 --models-dir-b 二选一）")
    ap.add_argument("--models-dir-a", type=str, help="模型目录 A；若与 --models-dir-b 同时给出，会先跑回测再对比")
    ap.add_argument("--models-dir-b", type=str, help="模型目录 B")
    ap.add_argument("--label-a", type=str, default=None, help="A 的显示名，默认从 models_dir 或 json 路径推断")
    ap.add_argument("--label-b", type=str, default=None, help="B 的显示名")
    ap.add_argument("--append-csv", type=str, default=None, help="追加到该 CSV，用于每日对比（列: date, label, total_trades, wins, losses, win_rate%%, profit_pct%%, max_drawdown%%）")
    args = ap.parse_args()

    # 若指定了 models-dir-a/b，先跑回测
    if args.models_dir_a and args.models_dir_b:
        today = datetime.now().strftime("%Y%m%d")
        reports = PROJECT_ROOT / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        ja = reports / f"compare_a_{today}.json"
        jb = reports / f"compare_b_{today}.json"
        ha = reports / f"compare_a_{today}.html"
        hb = reports / f"compare_b_{today}.html"
        print("运行回测 套A ...")
        data_a = _run_backtest(args.models_dir_a, ja, ha)
        print("运行回测 套B ...")
        data_b = _run_backtest(args.models_dir_b, jb, hb)
        label_a = args.label_a or Path(args.models_dir_a).name
        label_b = args.label_b or Path(args.models_dir_b).name
    elif args.json_a and args.json_b:
        data_a = json.loads(Path(args.json_a).read_text(encoding="utf-8"))
        data_b = json.loads(Path(args.json_b).read_text(encoding="utf-8"))
        label_a = args.label_a or Path(args.json_a).stem
        label_b = args.label_b or Path(args.json_b).stem
    else:
        print("请提供 (--json-a 与 --json-b) 或 (--models-dir-a 与 --models-dir-b)")
        sys.exit(1)

    agg_a = _aggregate(data_a)
    agg_b = _aggregate(data_b)

    # 打印对比
    print("\n" + "=" * 70)
    print("两套模型回测对比")
    print("=" * 70)
    print(f"  {'指标':<16} {'A: ' + label_a:<24} {'B: ' + label_b:<24}")
    print("-" * 70)
    print(f"  {'总交易次数':<16} {agg_a['total_trades']:<24} {agg_b['total_trades']:<24}")
    print(f"  {'胜 / 负':<16} {agg_a['wins']} / {agg_a['losses']:<18} {agg_b['wins']} / {agg_b['losses']:<18}")
    print(f"  {'胜率 %':<16} {agg_a['win_rate']:.1f}%{'':<20} {agg_b['win_rate']:.1f}%")
    print(f"  {'盈亏 %':<16} {agg_a['profit_pct']:+.1f}%{'':<20} {agg_b['profit_pct']:+.1f}%")
    print(f"  {'最大回撤 %':<16} {agg_a['max_drawdown']:.1f}%{'':<20} {agg_b['max_drawdown']:.1f}%")
    print("=" * 70)
    better = "A" if agg_a["profit_pct"] > agg_b["profit_pct"] else "B"
    print(f"  按盈亏% 本轮更优: {label_a if better == 'A' else label_b}")
    print()

    # 追加 CSV
    if args.append_csv:
        csv_path = Path(args.append_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        date = datetime.now().strftime("%Y-%m-%d")
        exists = csv_path.exists()
        with open(csv_path, "a", encoding="utf-8") as f:
            if not exists:
                f.write("date,label,total_trades,wins,losses,win_rate_pct,profit_pct,max_drawdown_pct\n")
            f.write(f"{date},{label_a},{agg_a['total_trades']},{agg_a['wins']},{agg_a['losses']},{agg_a['win_rate']:.1f},{agg_a['profit_pct']:.1f},{agg_a['max_drawdown']:.1f}\n")
            f.write(f"{date},{label_b},{agg_b['total_trades']},{agg_b['wins']},{agg_b['losses']},{agg_b['win_rate']:.1f},{agg_b['profit_pct']:.1f},{agg_b['max_drawdown']:.1f}\n")
        print(f"已追加到 {csv_path}")


if __name__ == "__main__":
    main()
