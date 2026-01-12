"""预测器抽象与 LITETime 适配器模块。

定义统一的预测器接口，并提供 aeon LITETimeClassifier 的适配实现。
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BasePredictor(ABC):
    """预测器抽象基类。

    定义统一的预测器接口，支持不同的模型实现。
    """

    @abstractmethod
    def load(self, num_features: int) -> None:
        """加载模型。

        Args:
            num_features: 输入特征维度。
        """

    @abstractmethod
    def predict(self, batch: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """执行批量预测。

        Args:
            batch: 形状为 (batch_size, seq_len, n_features) 的输入数据。
            lengths: 形状为 (batch_size,) 的实际序列长度。

        Returns:
            (predictions, probabilities)
            - predictions: 形状为 (batch_size,) 的类别索引。
            - probabilities: 形状为 (batch_size, n_classes) 的类别概率。
        """


class LITETimePredictor(BasePredictor):
    """aeon LITETimeClassifier 的预测器适配器。

    适配 aeon.classification.deep_learning.LITETimeClassifier 到统一接口。
    """

    def __init__(
        self,
        model_path: str | list[str],
        classes: np.ndarray | list[str],
    ):
        """初始化 LITETime 预测器。

        Args:
            model_path: 模型文件路径（单个或列表）。
            classes: 类别名称数组。
        """
        self.model_path = model_path
        self.classes = np.asarray(classes)
        self.model = None
        self._num_features = None

    def load(self, num_features: int) -> None:
        """加载 LITETime 模型。

        Args:
            num_features: 输入特征维度（用于验证）。
        """
        from aeon.classification.deep_learning import LITETimeClassifier

        self._num_features = num_features

        if isinstance(self.model_path, str):
            model_paths = [self.model_path]
        else:
            model_paths = list(self.model_path)

        self.model = LITETimeClassifier.load_model(
            model_path=model_paths,
            classes=self.classes,
        )

    def predict(self, batch: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """执行批量预测。

        Args:
            batch: 形状为 (batch_size, seq_len, n_features) 的输入数据。
            lengths: 形状为 (batch_size,) 的实际序列长度（LITETime 会自动处理）。

        Returns:
            (predictions, probabilities)
            - predictions: 形状为 (batch_size,) 的类别索引。
            - probabilities: 形状为 (batch_size, n_classes) 的类别概率。
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load()")

        # LITETime 的 predict 返回类别标签
        labels = self.model.predict(batch)

        # 将标签转换为索引
        predictions = np.array([
            np.where(self.classes == label)[0][0] if label in self.classes else 0
            for label in labels
        ])

        # 获取概率
        probabilities = self.model.predict_proba(batch)

        return predictions, probabilities


class PredictorFactory:
    """预测器工厂类。

    根据配置创建对应的预测器实例。
    """

    @staticmethod
    def create_from_meta(meta_path: str, model_dir: str = "") -> LITETimePredictor:
        """从 meta.json 文件创建 LITETime 预测器。

        Args:
            meta_path: meta.json 文件路径。
            model_dir: 模型目录（如果 meta 中的路径是相对路径）。

        Returns:
            配置好的 LITETimePredictor 实例。
        """
        from pathlib import Path

        import sys

        # 添加 radar_litetime 到路径以导入 data_utils
        project_root = Path(__file__).resolve().parents[2]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from data_utils import load_json

        meta = load_json(meta_path)

        # 获取类别名称
        class_names = meta.get("class_names", [])
        if not class_names:
            raise ValueError(f"meta.json 中缺少 class_names: {meta_path}")

        # 获取模型文件
        model_files = meta.get("model_files", [])
        if not model_files:
            raise ValueError(f"meta.json 中缺少 model_files: {meta_path}")

        # 解析模型路径
        if model_dir:
            model_dir_path = Path(model_dir)
            model_paths = []
            for f in model_files:
                candidate = Path(f)
                if candidate.is_file():
                    model_paths.append(candidate)
                else:
                    model_paths.append(model_dir_path / f)
        else:
            meta_dir = Path(meta_path).parent
            model_paths = [meta_dir / f for f in model_files]

        return LITETimePredictor(
            model_path=[str(p) for p in model_paths],
            classes=class_names,
        )

    @staticmethod
    def create(
        model_files: str | list[str],
        classes: list[str],
        model_dir: str = "",
    ) -> LITETimePredictor:
        """直接创建 LITETime 预测器。

        Args:
            model_files: 模型文件路径（单个或列表）。
            classes: 类别名称列表。
            model_dir: 模型目录（如果 model_files 中的路径是相对路径）。

        Returns:
            配置好的 LITETimePredictor 实例。
        """
        if isinstance(model_files, str):
            model_files = [f.strip() for f in model_files.split(",") if f.strip()]

        if model_dir:
            model_dir_path = Path(model_dir)
            model_paths = []
            for f in model_files:
                candidate = Path(f)
                if candidate.is_file():
                    model_paths.append(candidate)
                else:
                    model_paths.append(model_dir_path / f)
        else:
            model_paths = [Path(f) for f in model_files]

        return LITETimePredictor(
            model_path=[str(p) for p in model_paths],
            classes=classes,
        )
