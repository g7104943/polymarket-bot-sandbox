"""模型训练与预测流程测试（需有 data/raw 或 mock）"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_imports():
    from src.python.model_trainer import _load_lightgbm_params, train_one
    from src.python.predictor import load_predictor
    params = _load_lightgbm_params()
    assert "objective" in params or "verbosity" in params


def test_train_one_smoke():
    import pandas as pd
    import numpy as np
    from src.python.model_trainer import train_one, _load_lightgbm_params

    np.random.seed(42)
    n = 300
    X = pd.DataFrame(np.random.randn(n, 5).cumsum(axis=0) + 100, columns=[f"f{i}" for i in range(5)])
    y = pd.Series((np.random.rand(n) > 0.5).astype(int))
    result = train_one(X, y, params=_load_lightgbm_params(), val_ratio=0.2)
    model, metrics = result[0], result[1]
    assert hasattr(model, "predict")
    assert 0 <= metrics.get("accuracy", 0) <= 1
