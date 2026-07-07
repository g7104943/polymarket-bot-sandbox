"""特征工程测试"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.python.feature_engineering import build_features, build_labels, prepare_train_data, get_feature_columns


def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.random.rand(n) * 2
    low = close - np.random.rand(n) * 2
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.2,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


def test_build_features():
    df = _make_ohlcv(150)
    out = build_features(df)
    assert "kdj_k" in out.columns and "rang_osc" in out.columns
    assert "macd_line" in out.columns and "obv" in out.columns
    assert "rsi" in out.columns and "bb_upper" in out.columns


def test_build_labels():
    df = _make_ohlcv(50)
    y = build_labels(df)
    assert len(y) == len(df)
    assert set(y.dropna().unique()).issubset({0, 1})


def test_prepare_train_data():
    df = _make_ohlcv(200)
    X, y = prepare_train_data(df)
    assert len(X) == len(y)
    assert len(X) >= 50
    assert X.notna().all().all() and y.notna().all()
