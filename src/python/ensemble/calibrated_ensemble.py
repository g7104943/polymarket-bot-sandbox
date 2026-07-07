"""
Calibrated Ensemble + Stacking + 不确定性估计（步骤 8）

基模型组合：
- 方案 A 短期模型（90 天滚动窗口）—— 适应当前市场
- 方案 B 长期模型（分层采样 2 年）—— 应对罕见事件
- 最优 Encoder（GRU 或 Transformer）

Stacking Meta-Model：L2 正则化 Logistic Regression
不确定性估计：基于多模型预测的 std

用法：
    from src.python.ensemble.calibrated_ensemble import CalibratedEnsemble
    ens = CalibratedEnsemble()
    ens.add_model("short_term", model_a, calibrator_a, "isotonic")
    ens.add_model("long_term", model_b, calibrator_b, "sigmoid")
    prob, uncertainty = ens.predict(X)
"""

import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression


class CalibratedEnsemble:
    """
    校准后的集成模型。

    流程：
    1. 每个基模型独立预测 P(UP)
    2. 对每个基模型的输出应用各自的 calibrator
    3. 校准后的概率作为 meta 特征输入 stacking meta-model
    4. Meta-model 输出最终 P(UP)
    5. 多模型预测的 std 作为不确定性估计
    """

    def __init__(self):
        self.base_models: Dict[str, Dict[str, Any]] = {}
        self.meta_model: Optional[LogisticRegression] = None
        self.meta_feature_order: List[str] = []

    def add_model(
        self,
        name: str,
        model: Any,
        calibrator: Any = None,
        calibration_method: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
    ):
        """
        添加一个基模型。

        参数：
            name: 模型名（如 "short_term", "long_term", "gru_encoder"）
            model: 模型对象（需有 predict 或 predict_proba 方法）
            calibrator: 校准器（可选）
            calibration_method: 校准方法
            feature_names: 该模型使用的特征列名
        """
        self.base_models[name] = {
            "model": model,
            "calibrator": calibrator,
            "calibration_method": calibration_method,
            "feature_names": feature_names,
        }

    def predict_base(self, X: np.ndarray, name: str) -> np.ndarray:
        """
        用单个基模型预测 P(UP)。

        返回校准后的概率数组。
        """
        info = self.base_models[name]
        model = info["model"]

        # 获取原始概率
        if hasattr(model, "predict_proba"):
            raw_probs = model.predict_proba(X)[:, 1]
        else:
            raw_probs = model.predict(X)
        raw_probs = np.asarray(raw_probs).ravel()

        # 应用校准
        if info["calibrator"] is not None and info["calibration_method"]:
            from ..predictor import apply_calibration
            raw_probs = apply_calibration(
                info["calibrator"], info["calibration_method"], raw_probs
            )

        return np.clip(np.asarray(raw_probs).ravel(), 0, 1)

    def predict_all_base(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """用所有基模型预测，返回 {name: probs}。"""
        return {name: self.predict_base(X, name) for name in self.base_models}

    def fit_meta_model(
        self,
        X_oof_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        C: float = 1.0,
    ):
        """
        训练 Stacking Meta-Model。

        参数：
            X_oof_predictions: {model_name: oof_probs} 各基模型的 Out-Of-Fold 预测
            y_true: 真实标签
            C: L2 正则化强度（默认 1.0）
        """
        self.meta_feature_order = sorted(X_oof_predictions.keys())
        meta_X = np.column_stack([
            X_oof_predictions[name] for name in self.meta_feature_order
        ])
        y = np.asarray(y_true).ravel()

        self.meta_model = LogisticRegression(
            C=C, max_iter=1000, solver="lbfgs", penalty="l2"
        )
        self.meta_model.fit(meta_X, y)

    def predict(
        self, X: np.ndarray, return_uncertainty: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        集成预测。

        参数：
            X: 特征矩阵
            return_uncertainty: 是否返回不确定性

        返回：
            (probs, uncertainty)
            probs: 最终 P(UP) 数组
            uncertainty: 不确定性估计（若 return_uncertainty=True）
        """
        base_preds = self.predict_all_base(X)

        if self.meta_model is not None and self.meta_feature_order:
            # 使用 stacking meta-model
            meta_X = np.column_stack([
                base_preds[name] for name in self.meta_feature_order
            ])
            probs = self.meta_model.predict_proba(meta_X)[:, 1]
        else:
            # 简单平均（fallback）
            all_probs = np.stack(list(base_preds.values()), axis=0)
            probs = all_probs.mean(axis=0)

        probs = np.clip(probs, 0, 1)

        uncertainty = None
        if return_uncertainty and len(base_preds) > 1:
            all_probs = np.stack(list(base_preds.values()), axis=0)
            std = all_probs.std(axis=0)
            max_std = std.max() + 1e-6
            uncertainty = std / max_std  # 归一化到 [0, 1]

        return probs, uncertainty

    def adjust_position_size(
        self,
        base_size: float,
        uncertainty: float,
        kelly_fraction: float = 0.5,
    ) -> float:
        """
        根据不确定性调整仓位大小。

        公式：adjusted = base_size * kelly_fraction * (1 - uncertainty)

        参数：
            base_size: 原始下注金额
            uncertainty: 不确定性（0-1，0=最确定）
            kelly_fraction: Kelly 比例（默认 0.5 = half Kelly）

        返回：
            调整后的下注金额
        """
        return base_size * kelly_fraction * (1 - float(uncertainty))

    def save(self, output_dir: Path):
        """保存 ensemble 到目录。"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存 meta-model
        if self.meta_model is not None:
            joblib.dump(self.meta_model, output_dir / "meta_model.pkl")

        # 保存基模型信息（不保存模型本身，只保存元信息）
        meta_info = {
            "base_models": list(self.base_models.keys()),
            "meta_feature_order": self.meta_feature_order,
            "n_base_models": len(self.base_models),
        }
        (output_dir / "ensemble_config.json").write_text(
            json.dumps(meta_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 保存每个基模型的 calibrator
        for name, info in self.base_models.items():
            if info["calibrator"] is not None:
                cal_path = output_dir / f"calibrator_{name}.joblib"
                joblib.dump(info["calibrator"], cal_path)

    @classmethod
    def load(cls, model_dir: Path) -> "CalibratedEnsemble":
        """从目录加载 ensemble。"""
        model_dir = Path(model_dir)
        ens = cls()

        config_path = model_dir / "ensemble_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            ens.meta_feature_order = config.get("meta_feature_order", [])

        meta_path = model_dir / "meta_model.pkl"
        if meta_path.exists():
            ens.meta_model = joblib.load(meta_path)

        return ens


def estimate_uncertainty(base_predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    从多个基模型的预测估计不确定性。

    参数：
        base_predictions: (n_models, n_samples) 各模型的 P(UP)

    返回：
        (mean_pred, normalized_uncertainty)
    """
    mean_pred = np.mean(base_predictions, axis=0)
    std_pred = np.std(base_predictions, axis=0)
    max_std = std_pred.max() + 1e-6
    normalized_uncertainty = std_pred / max_std
    return mean_pred, normalized_uncertainty
