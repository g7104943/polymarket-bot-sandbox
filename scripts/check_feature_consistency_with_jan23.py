#!/usr/bin/env python3
"""
检查当前 build_features + add_multi_timeframe_features 与 Jan 23 模型期望的特征是否一致。
- 名称齐全、预测时用 full[feature_names] 按模型顺序取列 → 至少「合约」一致。
- 当前管线若多出特征（模型未用）或产出顺序与当时不同，说明 feature_engineering 有过改动；
  公式/窗口若改过，同一根 K 线算出的特征值可能和当时不同 → 输出从 80%+ 变成 50% 多。
- 有缺 → 预测会报错；有多/顺序不同 → 不报错但可能数值已变。
"""
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.python.data_fetcher import load_ohlcv
from src.python.feature_engineering import build_features, add_multi_timeframe_features

def main():
    model_dir = PROJECT_ROOT / "data" / "models_C" / "lightgbm_XRP_USDT_15m_20260123_152549"
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        print("❌ 未找到 Jan 23 模型 metadata:", meta_path)
        return

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_feature_names = meta.get("feature_names", [])
    print(f"Jan 23 模型期望特征数: {len(model_feature_names)}")
    print()

    # 当前管线：XRP 15m 数据 → build_features → add_multi_timeframe
    df = load_ohlcv("XRP/USDT", "15m")
    if df.empty or len(df) < 200:
        print("❌ XRP 15m 数据不足，无法计算特征")
        return

    df = build_features(df, "XRP/USDT")
    df = add_multi_timeframe_features(df, "XRP/USDT")
    non_ohlcv = {"open", "high", "low", "close", "volume", "timestamp", "date", "_tm"}
    current_feature_set = set(c for c in df.columns if c not in non_ohlcv)
    model_set = set(model_feature_names)

    missing = model_set - current_feature_set
    extra = current_feature_set - model_set

    if missing:
        print("❌ 当前管线缺少以下特征（模型需要）:")
        for f in sorted(missing)[:20]:
            print(f"   - {f}")
        if len(missing) > 20:
            print(f"   ... 共 {len(missing)} 个")
        print()
    else:
        print("✅ 当前管线包含模型所需的全部特征名")
    if extra:
        print("⚠️ 当前管线多出以下特征（模型未使用，预测时不会用到）:")
        for f in sorted(extra)[:15]:
            print(f"   - {f}")
        if len(extra) > 15:
            print(f"   ... 共 {len(extra)} 个")
        print()
    else:
        print("✅ 当前管线没有多出模型未知的特征")

    # 顺序：预测时用 full[feature_names]，顺序以模型为准，所以只要名字齐全即可
    if not missing:
        order_ok = list(df.columns) == model_feature_names
        if not order_ok:
            # 检查当前 build_features 产出顺序是否与模型一致（前若干项）
            first_current = [c for c in df.columns if c in model_feature_names][:20]
            first_model = model_feature_names[:20]
            if first_current != first_model:
                print("⚠️ 特征顺序与 Jan 23 模型不一致（预测时会用 model 的 feature_names 取列，一般不影响结果）")
                print("   模型前 5 个:", first_model[:5])
                print("   当前前 5 个:", first_current[:5])
            else:
                print("✅ 特征顺序与模型一致（前 20 项）")
    print()

    # Git：1 月 23 日后是否改过 feature_engineering
    try:
        r = subprocess.run(
            ["git", "log", "--oneline", "--since=2026-01-23", "--", "src/python/feature_engineering.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            print("📌 2026-01-23 之后 feature_engineering.py 有提交:")
            for line in r.stdout.strip().splitlines()[:10]:
                print(f"   {line}")
            print("   → 若公式/窗口有改动，同一根 K 线算出的特征值可能和当时不同，会导致输出从 80%+ 变成 50% 多。")
        else:
            print("📌 2026-01-23 之后 feature_engineering.py 无提交（或非 git 仓库）")
    except Exception as e:
        print("📌 无法执行 git log:", e)

if __name__ == "__main__":
    main()
