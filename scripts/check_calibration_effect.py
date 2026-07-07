#!/usr/bin/env python3
"""
检测校准是否把概率压向 0.5，导致双阈值/单阈值组合下单变少。
与模拟逻辑完全一致：同一数据加载、特征构建、模型加载、raw vs 校准后概率比较。

用法（项目根目录）:
  python3 scripts/check_calibration_effect.py                    # 只跑 logs_xrp_20_80，默认全数据
  python3 scripts/check_calibration_effect.py --all-old-combos   # 跑 5 个旧模型组合，默认全数据
  python3 scripts/check_calibration_effect.py --last-n 6000      # 用最近 N 根 K 线（0=全数据）
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.python.predictor import (
    load_predictor,
    apply_calibration,
    _find_symbol_timeframe_model,
)
from src.python.data_fetcher import load_ohlcv, update_latest
from src.python.feature_engineering import build_features, add_multi_timeframe_features


# 与 启动三个版本.sh、回测报告一致的 5 个旧模型组合（仅 15m）：eth_92、eth_10-90、btc_55无动态、xrp_53、xrp_20-80
OLD_COMBOS = [
    {"name": "logs_eth", "models_dir": "data/models", "symbol": "ETH/USDT", "tf": "15m", "single": 0.92, "up": None, "down": None},
    {"name": "logs_eth_10_90", "models_dir": "data/models", "symbol": "ETH/USDT", "tf": "15m", "single": None, "up": 0.9, "down": 0.1},
    {"name": "logs_btc", "models_dir": "data/models_C", "symbol": "BTC/USDT", "tf": "15m", "single": 0.55, "up": None, "down": None},
    {"name": "logs_xrp", "models_dir": "data/models_C", "symbol": "XRP/USDT", "tf": "15m", "single": 0.53, "up": None, "down": None},
    {"name": "logs_xrp_20_80", "models_dir": "data/models_C", "symbol": "XRP/USDT", "tf": "15m", "single": None, "up": 0.8, "down": 0.2},
]


def _pass_threshold(prob: float, pred: int, single: float, up: float, down: float) -> bool:
    """与模拟/实盘一致：单阈值 effective_prob>=t；双阈值 UP 时 prob>=up，DOWN 时 prob<down。"""
    eff = prob if pred == 1 else (1.0 - prob)
    if single is not None:
        return eff >= single
    if pred == 1:
        return prob >= up
    return prob < down


def run_detection(
    models_dir: Path,
    symbol: str,
    timeframe: str,
    single: float,
    up: float,
    down: float,
    last_n: int = 0,
) -> dict:
    """
    与 prediction_writer 完全一致：同数据源、同特征、同模型、先 raw 再校准。
    last_n<=0 时用全数据；否则对最近 last_n 根 K 线逐行算 raw 与 calibrated prob，统计「达阈值次数」。
    """
    root = PROJECT_ROOT
    if not models_dir.is_absolute():
        models_dir = root / models_dir
    model_dir = _find_symbol_timeframe_model(symbol, timeframe, models_root=models_dir)
    if not model_dir:
        return {"error": f"未找到模型 {symbol} {timeframe} in {models_dir}"}
    model, feats, meta = load_predictor(model_dir)
    calibrator = meta.get("calibrator")
    cal_method = meta.get("calibration")

    # 全数据：用 load_ohlcv 全部行（20万+）；与 prediction_writer 同源，build_features，需要时 add_multi_timeframe_features
    df = load_ohlcv(symbol, timeframe)
    if df.empty or len(df) < 100:
        return {"error": f"数据不足 {symbol} {timeframe} ({len(df)} 行)"}
    # 不再截断为 8000，全数据即全部 K 线
    df = build_features(df, symbol)
    if any((f or "").startswith("mtf_") for f in (feats or [])):
        try:
            update_latest(symbol, "1h")
            update_latest(symbol, "4h")
        except Exception:
            pass
        df = add_multi_timeframe_features(df, symbol)
    missing = [f for f in feats if f not in df.columns]
    if missing:
        return {"error": f"缺特征 {missing[:5]}"}
    filled = df[feats].ffill().bfill()
    if filled.isna().any().any():
        return {"error": "特征含 NaN"}
    n = len(filled)
    start_i = 0 if last_n <= 0 else max(0, n - last_n)

    raw_pass = 0
    cal_pass = 0
    raw_list = []
    cal_list = []
    for i in range(start_i, n):
        row = filled.iloc[[i]]
        if hasattr(model, "predict_proba"):
            raw_prob = float(model.predict_proba(row)[0, 1])
        else:
            raw_prob = float(model.predict(row)[0])
        raw_pred = 1 if raw_prob >= 0.5 else 0
        cal_prob = raw_prob
        if calibrator is not None and cal_method:
            cal_prob = apply_calibration(calibrator, cal_method, raw_prob)
            if hasattr(cal_prob, "item"):
                cal_prob = float(cal_prob)
        cal_pred = 1 if cal_prob >= 0.5 else 0
        if _pass_threshold(raw_prob, raw_pred, single, up, down):
            raw_pass += 1
        if _pass_threshold(cal_prob, cal_pred, single, up, down):
            cal_pass += 1
        raw_list.append(raw_prob)
        cal_list.append(cal_prob)

    return {
        "has_calibrator": calibrator is not None and bool(cal_method),
        "cal_method": cal_method,
        "n_samples": n - start_i,
        "raw_pass": raw_pass,
        "cal_pass": cal_pass,
        "raw_min": min(raw_list),
        "raw_max": max(raw_list),
        "cal_min": min(cal_list),
        "cal_max": max(cal_list),
    }


def main():
    ap = argparse.ArgumentParser(description="检测校准是否压缩概率（与模拟逻辑一致）")
    ap.add_argument("--all-old-combos", action="store_true", help="检测全部 5 个旧模型组合")
    ap.add_argument("--last-n", type=int, default=0, help="用最近 N 根 K 线统计，0 表示全数据，默认 0")
    args = ap.parse_args()

    if args.all_old_combos:
        combos = OLD_COMBOS
    else:
        combos = [c for c in OLD_COMBOS if c["name"] == "logs_xrp_20_80"]
        if not combos:
            combos = OLD_COMBOS[:1]

    last_n = args.last_n
    range_desc = "全数据" if last_n <= 0 else f"最近 {last_n} 根 K 线"
    print(f"检测校准效果（与模拟逻辑一致，{range_desc}）\n")
    for c in combos:
        models_dir = Path(c["models_dir"])
        res = run_detection(
            models_dir, c["symbol"], c["tf"],
            c["single"], c["up"], c["down"],
            last_n=args.last_n,
        )
        if "error" in res:
            print(f"  [{c['name']}] 错误: {res['error']}")
            continue
        thr_desc = f"单≥{c['single']}" if c["single"] is not None else f"UP≥{c['up']} 或 DOWN<{c['down']}"
        print(f"  [{c['name']}] 阈值: {thr_desc}")
        print(f"      有校准: {res['has_calibrator']} ({res.get('cal_method', '')})")
        print(f"      达阈值次数: raw={res['raw_pass']}  校准后={res['cal_pass']}  (共 {res['n_samples']} 样本)")
        print(f"      raw  P(UP) 范围: [{res['raw_min']:.3f}, {res['raw_max']:.3f}]")
        print(f"      校准 P(UP) 范围: [{res['cal_min']:.3f}, {res['cal_max']:.3f}]")
        if res["has_calibrator"] and res["raw_pass"] > res["cal_pass"]:
            print(f"      >>> 校准明显压缩了极端概率，可能导致下单变少")
        print()
    print("说明:")
    print("  - 若 raw 达阈值次数远大于校准后，说明校准把概率压向 0.5，可能导致下单变少。")
    print("  - 规则「有校准且次数变少→无校准」：raw>cal 则该组合在启动脚本中应设 SKIP_CALIBRATION=1；cal≥raw 则用校准（不设 SKIP）。")
    print("  - 仅 data/models_C 有校准；data/models（ETH 三组）无校准。")
    print("  - 仅对某一组合的 Python 进程设 SKIP_CALIBRATION=1 时，其他组合不受影响（各进程独立 env）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
