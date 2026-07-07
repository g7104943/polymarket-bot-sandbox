"""
Purged Walk-Forward 验证模块：
- 替代 sklearn TimeSeriesSplit，增加 purge/embargo 防止时间序列泄漏
- 在 train 末尾删除 purge 行，val 开头删除 embargo 行
- 支持按百分比或固定行数指定 purge/embargo
- 每个 fold 记录详细指标到 fold_metrics.json

用法：
    from src.python.validation.purged_walk_forward import PurgedWalkForward
    pwf = PurgedWalkForward(n_splits=5, purge_pct=0.01, embargo_pct=0.005)
    for train_idx, val_idx in pwf.split(X):
        ...
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Generator


class PurgedWalkForward:
    """
    Purged Walk-Forward 时间序列交叉验证。

    与 sklearn TimeSeriesSplit 的区别：
    1. 在每个 fold 的 train 末尾删除 purge 样本
    2. 在每个 fold 的 val 开头删除 embargo 样本
    3. 减少因时间序列自相关导致的信息泄漏

    参数：
        n_splits: 折数（默认 5）
        purge_pct: 从训练集末尾删除的比例（默认 0.01 = 1%）
        embargo_pct: 从验证集开头删除的比例（默认 0.005 = 0.5%）
        purge_bars: 固定删除的行数（与 purge_pct 二选一，优先级更高）
        embargo_bars: 固定删除的行数（与 embargo_pct 二选一，优先级更高）
        min_train_size: 训练集最小样本数（低于则跳过该 fold）
        min_val_size: 验证集最小样本数（低于则跳过该 fold）
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_pct: float = 0.01,
        embargo_pct: float = 0.005,
        purge_bars: Optional[int] = None,
        embargo_bars: Optional[int] = None,
        min_train_size: int = 100,
        min_val_size: int = 50,
    ):
        self.n_splits = n_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars
        self.min_train_size = min_train_size
        self.min_val_size = min_val_size

    def _compute_purge_embargo(self, n_train: int, n_val: int) -> Tuple[int, int]:
        """计算实际的 purge 和 embargo 行数。"""
        if self.purge_bars is not None:
            purge = self.purge_bars
        else:
            purge = max(0, int(n_train * self.purge_pct))

        if self.embargo_bars is not None:
            embargo = self.embargo_bars
        else:
            embargo = max(0, int(n_val * self.embargo_pct))

        return purge, embargo

    def split(
        self, X, y=None, groups=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        生成 purged walk-forward 的 (train_idx, val_idx) 对。

        参数：
            X: 特征矩阵（仅用 len(X) 确定样本数）
            y: 忽略（兼容 sklearn 接口）
            groups: 忽略

        生成：
            (train_indices, val_indices) 元组
        """
        n = len(X)
        if n < self.min_train_size + self.min_val_size:
            raise ValueError(
                f"样本数 {n} 不足以划分（需 >= {self.min_train_size + self.min_val_size}）"
            )

        # 与 sklearn TimeSeriesSplit 一致的分割逻辑
        # 每个 fold 的验证集大小约 n / (n_splits + 1)
        fold_size = n // (self.n_splits + 1)

        for i in range(self.n_splits):
            # 原始边界（与 TimeSeriesSplit 一致）
            train_end = fold_size * (i + 1) + fold_size  # 不含
            val_end = train_end + fold_size  # 不含
            if val_end > n:
                val_end = n

            raw_train_idx = np.arange(0, train_end)
            raw_val_idx = np.arange(train_end, val_end)

            if len(raw_train_idx) == 0 or len(raw_val_idx) == 0:
                continue

            # 应用 purge/embargo
            purge, embargo = self._compute_purge_embargo(
                len(raw_train_idx), len(raw_val_idx)
            )

            if purge > 0 and len(raw_train_idx) > purge:
                raw_train_idx = raw_train_idx[:-purge]

            if embargo > 0 and len(raw_val_idx) > embargo:
                raw_val_idx = raw_val_idx[embargo:]

            # 检查最小样本数
            if len(raw_train_idx) < self.min_train_size:
                continue
            if len(raw_val_idx) < self.min_val_size:
                continue

            yield raw_train_idx, raw_val_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """返回实际产生的 fold 数（可能因 min_size 少于 n_splits）。"""
        if X is None:
            return self.n_splits
        return sum(1 for _ in self.split(X))

    def __repr__(self) -> str:
        return (
            f"PurgedWalkForward(n_splits={self.n_splits}, "
            f"purge_pct={self.purge_pct}, embargo_pct={self.embargo_pct}, "
            f"purge_bars={self.purge_bars}, embargo_bars={self.embargo_bars})"
        )


def save_fold_metrics(
    fold_metrics: List[Dict[str, Any]],
    output_dir: Path,
    filename: str = "fold_metrics.json",
) -> Path:
    """
    保存每个 fold 的指标到 JSON 文件。

    参数：
        fold_metrics: 列表，每个元素是一个 fold 的指标字典
        output_dir: 输出目录
        filename: 文件名

    返回：
        写出的文件路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    # 确保 JSON 可序列化
    serializable = []
    for m in fold_metrics:
        sm = {}
        for k, v in m.items():
            if isinstance(v, (np.integer, np.int64)):
                sm[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                sm[k] = float(v)
            elif isinstance(v, np.ndarray):
                sm[k] = v.tolist()
            else:
                sm[k] = v
        serializable.append(sm)

    path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
