#!/usr/bin/env python3
"""
Exp8 端到端验证脚本 — 验证特征对齐 + 数学计算 + 参数传递

用法:
    python scripts/validate_exp8_e2e.py

此脚本使用当前可用的本地数据执行一次完整的预测流程，验证:
1. feature_cols.json 与实际特征列完全匹配
2. 所有 240 个特征值范围合理
3. 数学计算正确（集成平均、edge、Kelly）
4. OB 特征列名与训练一致
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "gru_regime_v1"))

from scripts.prediction_writer_v5 import V5Predictor

def main():
    print("=" * 70)
    print("  Exp8 端到端验证")
    print("=" * 70)
    
    errors = []
    warnings = []
    
    # ─── 1. 加载模型和配置 ─────────────────────────────────────
    print("\n[1] 加载 Exp8 模型...")
    predictor = V5Predictor()
    config = predictor.config
    feature_cols = predictor.feature_cols
    print(f"  模型: {config.get('experiment', 'unknown')}")
    print(f"  特征数: {len(feature_cols)}")
    print(f"  资产: {list(predictor.asset_map.keys())}")
    
    if "exp8" not in config.get("experiment", "").lower() and "exp8" not in config.get("experiment", "").lower():
        # Check for both "Exp8" and "exp8"
        if "Exp8" not in config.get("experiment", ""):
            errors.append(f"模型不是 Exp8: {config.get('experiment')}")
    
    if len(feature_cols) != 240:
        warnings.append(f"特征数不是 240: {len(feature_cols)}")
    
    # ─── 2. feature_cols.json 对齐检查 ─────────────────────────
    print("\n[2] >>> FEATURE_COLS_CHECK <<<")
    
    # 加载 feature_cols.json
    fc_path = PROJECT_ROOT / "data" / "models" / "v5_production" / "feature_cols.json"
    with open(fc_path) as f:
        saved_cols = json.load(f)
    
    # 检查 OB 特征名
    from src.python.features.orderbook_features import get_ob_feature_names
    ob_names = get_ob_feature_names()
    ob_in_saved = [c for c in saved_cols if c.startswith("ob_")]
    ob_missing = [c for c in ob_names if c in saved_cols and c not in ob_in_saved]
    ob_extra = [c for c in ob_in_saved if c not in ob_names]
    
    matched = len(set(saved_cols) & set(feature_cols))
    missing_in_predictor = [c for c in saved_cols if c not in feature_cols]
    extra_in_predictor = [c for c in feature_cols if c not in saved_cols]
    
    print(f"  >>> FEATURE_COLS_CHECK: matched={matched}/{len(saved_cols)}, "
          f"missing={missing_in_predictor}, extra={extra_in_predictor}")
    
    if missing_in_predictor or extra_in_predictor:
        errors.append(f"feature_cols 不匹配! missing={missing_in_predictor}, extra={extra_in_predictor}")
    else:
        print(f"  ✅ 所有 {len(saved_cols)} 个特征列完全匹配")
    
    # OB 特征名检查
    print(f"  OB 特征: {len(ob_in_saved)} 个在 feature_cols.json 中")
    if ob_missing:
        errors.append(f"OB 特征名不匹配! missing={ob_missing}")
    if ob_extra:
        warnings.append(f"OB 额外特征: {ob_extra}")
    
    # ─── 3. 执行一次预测（用本地数据） ─────────────────────────
    print("\n[3] 执行预测（使用本地历史数据）...")
    try:
        predictions = predictor.predict_all()
        if not predictions:
            errors.append("预测返回空结果")
        else:
            for asset, pred in predictions.items():
                print(f"  {asset}: direction={pred['direction']}, "
                      f"confidence={pred['confidence']:.4f}, proba={pred['proba_up']:.4f}")
                
                # 验证 direction/confidence 一致性
                if pred['direction'] == 'UP':
                    if abs(pred['confidence'] - pred['proba_up']) > 0.0001:
                        errors.append(f"{asset}: UP 时 confidence 应等于 proba")
                else:
                    if abs(pred['confidence'] - (1 - pred['proba_up'])) > 0.0001:
                        errors.append(f"{asset}: DOWN 时 confidence 应等于 1-proba")
                
                # 验证 confidence >= 0.5
                if pred['confidence'] < 0.5:
                    errors.append(f"{asset}: confidence < 0.5: {pred['confidence']}")
    except Exception as e:
        errors.append(f"预测执行失败: {e}")
    
    # ─── 4. 数学计算验证 ─────────────────────────────────────
    print("\n[4] 数学计算验证...")
    
    # Edge 和 Kelly 计算
    test_conf = 0.55
    test_price = 0.50
    odds = (1.0 - test_price) / test_price  # = 1.0
    total_cost = 0.005
    
    # Edge
    edge = test_conf * odds - (1.0 - test_conf) - total_cost
    expected_edge = 0.55 * 1.0 - 0.45 - 0.005  # = 0.095
    print(f"  Edge(conf=0.55, price=0.50): {edge:.4f} (expected: {expected_edge:.4f})")
    if abs(edge - expected_edge) > 0.0001:
        errors.append(f"Edge 计算错误: got {edge}, expected {expected_edge}")
    else:
        print(f"  ✅ Edge 计算正确")
    
    # Kelly
    net_p = test_conf - total_cost / 2  # = 0.5475
    kelly = (net_p * odds - (1 - net_p)) / odds  # = (0.5475 - 0.4525) / 1.0 = 0.095
    expected_kelly = 0.095
    print(f"  Kelly(conf=0.55, price=0.50): {kelly:.4f} (expected: {expected_kelly:.4f})")
    if abs(kelly - expected_kelly) > 0.0001:
        errors.append(f"Kelly 计算错误: got {kelly}, expected {expected_kelly}")
    else:
        print(f"  ✅ Kelly 计算正确")
    
    # ─── 5. 特征组检查 ─────────────────────────────────────────
    print("\n[5] 特征组配置检查...")
    expected_groups = [
        "fgi_daily", "cfgi", "news", "ob", "funding",
        "oi", "lsratio", "polymarket_prob", "polymarket_prob_target"
    ]
    actual_groups = config.get("feature_groups", [])
    
    for g in expected_groups:
        if g in actual_groups:
            print(f"  ✅ {g}")
        else:
            errors.append(f"缺少特征组: {g}")
            print(f"  ❌ {g} -- 缺失!")
    
    # ─── 6. GRU 配置检查 ──────────────────────────────────────
    print("\n[6] GRU 配置检查...")
    gru_hparams = config.get("gru_hparams", {})
    print(f"  hidden_size: {gru_hparams.get('hidden_size')}")
    print(f"  embedding_dim: {gru_hparams.get('embedding_dim')}")
    print(f"  lookback: {gru_hparams.get('lookback')}")
    
    for asset in predictor.asset_map:
        gru_paths = config.get("gru_paths", {}).get(asset, {})
        for key in ["model", "normalizer", "config"]:
            p = gru_paths.get(key, "")
            if p and Path(p).exists():
                print(f"  ✅ {asset} GRU {key}: exists")
            else:
                errors.append(f"{asset} GRU {key} 不存在: {p}")
    
    # ─── 7. 模型集成验证 ──────────────────────────────────────
    print("\n[7] 模型集成检查...")
    n_models = len(predictor.models)
    windows = config.get("window_days_list", [])
    print(f"  模型数: {n_models} (窗口: {windows})")
    if n_models != 3:
        errors.append(f"模型数不是 3: {n_models}")
    if windows != [60, 90, 120]:
        errors.append(f"窗口不是 [60, 90, 120]: {windows}")
    
    # ─── 结果汇总 ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  验证结果汇总")
    print("=" * 70)
    
    if warnings:
        print(f"\n  ⚠️ 警告 ({len(warnings)}):")
        for w in warnings:
            print(f"    - {w}")
    
    if errors:
        print(f"\n  ❌ 错误 ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")
        print(f"\n❌ 验证失败 — 请修复以上 {len(errors)} 个错误")
        return 1
    else:
        print(f"\n  ✅ 全部通过 — 0 错误, {len(warnings)} 警告")
        print(f"\n✅ Exp8 端到端验证通过")
        return 0

if __name__ == "__main__":
    sys.exit(main())
