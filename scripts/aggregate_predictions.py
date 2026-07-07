"""
聚合 12 个预测市场（4 币 × 3 周期）的 up/down，供 Polymarket「高概率 up 就买」使用。

输入：3 个推理结果（15m / 1h / 4h 各 4 个 pair），可来自：
  - 3 个 JSON 文件（由各进程或单独推理脚本写入）
  - 独立 LightGBM predictor 的 API
输出：合并后的预测，按 prob 降序，支持 top-K 过滤，便于取「高概率」下注。

功能：
  - 概率阈值过滤：只保留 P(方向) >= prob_threshold 的预测
  - top-K 过滤：只保留概率最高的 K 个预测（避免资金分散）
  - 方向过滤：可选只保留 UP 或 DOWN 方向

用法：
  python scripts/aggregate_predictions.py --15m pred_15m.json --1h pred_1h.json --4h pred_4h.json
  python scripts/aggregate_predictions.py --15m pred_15m.json --top-k 5  # 只取 top 5 高概率
  python scripts/aggregate_predictions.py --config config/polymarket_threshold.json  # 从配置文件读取参数
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Any

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
TIMEFRAMES = ["15m", "1h", "4h"]

# 配置文件路径
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "polymarket_threshold.json"


def load_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_threshold_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载阈值配置文件"""
    path = Path(config_path) if config_path else CONFIG_PATH
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "prob_threshold": 0.55,
        "do_predict_min": 0,
        "top_k": None,
        "direction_filter": None,
    }


def merge_predictions(pred_15m: list, pred_1h: list, pred_4h: list) -> list:
    """将 3 组（各 4 条）合并为 12 条，每条含 pair, timeframe, direction, prob(可选)。"""
    out = []
    for name, lst in [("15m", pred_15m), ("1h", pred_1h), ("4h", pred_4h)]:
        for p in lst:
            p = dict(p)  # 复制避免修改原数据
            p["timeframe"] = name
            out.append(p)
    return out


def get_effective_prob(x: dict) -> float:
    """
    获取有效的概率值（用于排序和过滤）。
    对于 UP 方向，prob 本身就是 P(UP)
    对于 DOWN 方向，需要用 1 - prob 来表示 P(DOWN)
    """
    p = x.get("prob")
    if p is None or p == "":
        return 0.0
    try:
        pv = float(p)
        d = (x.get("direction") or "").lower()
        if d == "down":
            return 1.0 - pv  # P(DOWN) = 1 - P(UP)
        return pv  # P(UP)
    except (TypeError, ValueError):
        return 0.0


def filter_by_prob_threshold(merged: list, prob_threshold: float, do_predict_min: int) -> list:
    """
    概率阈值过滤：只保留 P(方向) >= prob_threshold 的预测。
    """
    out = []
    for x in merged:
        # do_predict 过滤
        if "do_predict" in x and x.get("do_predict") is not None:
            if x["do_predict"] < do_predict_min:
                continue
        
        # 概率过滤
        eff_prob = get_effective_prob(x)
        if eff_prob < prob_threshold:
            continue
        
        # 添加有效概率字段，便于后续排序
        x["effective_prob"] = round(eff_prob, 4)
        out.append(x)
    
    return out


def filter_by_direction(merged: list, direction_filter: Optional[str]) -> list:
    """方向过滤：只保留指定方向（UP 或 DOWN）的预测。"""
    if not direction_filter:
        return merged
    
    target = direction_filter.lower()
    return [x for x in merged if (x.get("direction") or "").lower() == target]


def filter_top_k(merged: list, top_k: Optional[int]) -> list:
    """
    Top-K 过滤：只保留概率最高的 K 个预测。
    按 effective_prob 降序排序后取前 K 个。
    """
    if not top_k or top_k <= 0:
        return merged
    
    # 按有效概率降序排序
    sorted_list = sorted(
        merged, 
        key=lambda x: x.get("effective_prob", get_effective_prob(x)),
        reverse=True
    )
    
    return sorted_list[:top_k]


def sort_predictions(merged: list) -> list:
    """
    排序预测结果：按有效概率降序。
    """
    return sorted(
        merged,
        key=lambda x: x.get("effective_prob", get_effective_prob(x)),
        reverse=True
    )


def aggregate_and_filter(
    pred_15m: list,
    pred_1h: list,
    pred_4h: list,
    prob_threshold: float = 0.55,
    do_predict_min: int = 0,
    top_k: Optional[int] = None,
    direction_filter: Optional[str] = None,
) -> list:
    """
    聚合并过滤预测结果。
    
    参数:
        pred_15m, pred_1h, pred_4h: 各周期的预测列表
        prob_threshold: 概率阈值（P(方向) >= 此值才保留）
        do_predict_min: do_predict 最小值过滤
        top_k: 只保留概率最高的 K 个预测（None 表示不限制）
        direction_filter: 方向过滤（"UP", "DOWN", 或 None）
    
    返回:
        过滤并排序后的预测列表
    """
    # 1. 合并
    merged = merge_predictions(pred_15m, pred_1h, pred_4h)
    
    # 2. 方向过滤（如果指定）
    if direction_filter:
        merged = filter_by_direction(merged, direction_filter)
    
    # 3. 概率阈值过滤
    merged = filter_by_prob_threshold(merged, prob_threshold, do_predict_min)
    
    # 4. 排序
    merged = sort_predictions(merged)
    
    # 5. Top-K 过滤
    if top_k:
        merged = filter_top_k(merged, top_k)
    
    return merged


def print_summary(merged: list, top_k: Optional[int] = None):
    """打印预测摘要"""
    if not merged:
        print("没有符合条件的预测")
        return
    
    print(f"\n{'='*60}")
    print(f"预测汇总 (共 {len(merged)} 条)")
    if top_k:
        print(f"Top-K 过滤: 保留概率最高的 {top_k} 个")
    print(f"{'='*60}")
    
    up_count = sum(1 for x in merged if (x.get("direction") or "").lower() == "up")
    down_count = len(merged) - up_count
    print(f"UP: {up_count}, DOWN: {down_count}")
    
    print(f"\n{'排名':<4} {'币对':<12} {'周期':<6} {'方向':<6} {'概率':<8} {'有效概率':<10}")
    print("-" * 60)
    
    for i, x in enumerate(merged, 1):
        pair = x.get("pair", "N/A")
        tf = x.get("timeframe", "N/A")
        direction = x.get("direction", "N/A")
        prob = x.get("prob", "N/A")
        eff_prob = x.get("effective_prob", "N/A")
        
        if isinstance(prob, (int, float)):
            prob = f"{prob:.4f}"
        if isinstance(eff_prob, (int, float)):
            eff_prob = f"{eff_prob:.4f}"
        
        print(f"{i:<4} {pair:<12} {tf:<6} {direction:<6} {prob:<8} {eff_prob:<10}")


def main():
    ap = argparse.ArgumentParser(
        description="聚合 15m/1h/4h 的预测，支持 top-K 高概率过滤",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python aggregate_predictions.py --15m pred_15m.json --1h pred_1h.json --4h pred_4h.json
  
  # 只取 top 5 高概率预测
  python aggregate_predictions.py --15m pred_15m.json --1h pred_1h.json --4h pred_4h.json --top-k 5
  
  # 只看 UP 方向的 top 3
  python aggregate_predictions.py --15m pred_15m.json --top-k 3 --direction up
  
  # 从配置文件读取参数
  python aggregate_predictions.py --15m pred_15m.json --config config/polymarket_threshold.json
        """
    )
    ap.add_argument("--15m", dest="f_15m", help="15m 预测 JSON 路径")
    ap.add_argument("--1h", dest="f_1h", help="1h 预测 JSON 路径")
    ap.add_argument("--4h", dest="f_4h", help="4h 预测 JSON 路径")
    ap.add_argument("--prob-threshold", type=float, default=None, 
                    help="P(方向)>=此值才保留，默认 0.55")
    ap.add_argument("--do-predict-min", type=int, default=None, 
                    help="do_predict>=此值才保留，默认 0")
    ap.add_argument("--top-k", type=int, default=None, 
                    help="只保留概率最高的 K 个预测（避免资金分散）")
    ap.add_argument("--direction", type=str, default=None, choices=["up", "down", "UP", "DOWN"],
                    help="只保留指定方向的预测")
    ap.add_argument("--config", type=str, default=None,
                    help="配置文件路径（覆盖命令行参数）")
    ap.add_argument("-o", "--out", default="-", help="输出 JSON 路径，默认 stdout")
    ap.add_argument("--summary", action="store_true", help="打印预测摘要")
    args = ap.parse_args()

    # 加载配置
    config = load_threshold_config(args.config)
    
    # 命令行参数覆盖配置文件
    prob_threshold = args.prob_threshold if args.prob_threshold is not None else config.get("prob_threshold", 0.55)
    do_predict_min = args.do_predict_min if args.do_predict_min is not None else config.get("do_predict_min", 0)
    top_k = args.top_k if args.top_k is not None else config.get("top_k")
    direction_filter = args.direction if args.direction else config.get("direction_filter")

    # 加载预测数据
    pred_15m = load_json(args.f_15m) if args.f_15m else _placeholder("15m")
    pred_1h = load_json(args.f_1h) if args.f_1h else _placeholder("1h")
    pred_4h = load_json(args.f_4h) if args.f_4h else _placeholder("4h")

    # 聚合和过滤
    merged = aggregate_and_filter(
        pred_15m, pred_1h, pred_4h,
        prob_threshold=prob_threshold,
        do_predict_min=do_predict_min,
        top_k=top_k,
        direction_filter=direction_filter,
    )

    # 打印摘要
    if args.summary:
        print_summary(merged, top_k)

    # 输出结果
    buf = json.dumps(merged, indent=2, ensure_ascii=False)
    if args.out == "-":
        if not args.summary:  # 避免混合输出
            print(buf)
    else:
        Path(args.out).write_text(buf, encoding="utf-8")
        print(f"结果已保存到: {args.out}")


def _placeholder(tf: str) -> list:
    """生成占位数据"""
    return [{"pair": p, "timeframe": tf, "direction": None, "prob": None} for p in PAIRS]


if __name__ == "__main__":
    main()
