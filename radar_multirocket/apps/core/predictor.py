"""预测器抽象与 MultiRocket 适配器模块。

定义统一的预测器接口，并提供 aeon MultiRocketClassifier 的适配实现。
"""

import joblib
from abc import ABC, abstractmethod
from pathlib import Path
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
            num_features: 输入特征维度（MultiRocket 忽略此参数）。
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


class MultiRocketPredictor(BasePredictor):
    """aeon MultiRocketClassifier 的预测器适配器。

    适配 aeon.classification.convolution_based.MultiRocketClassifier 到统一接口。
    """

    def __init__(
        self,
        model_path: str,
        classes: np.ndarray | list[str],
    ):
        """初始化 MultiRocket 预测器。

        Args:
            model_path: 模型文件路径（joblib 格式）。
            classes: 类别名称数组。
        """
        self.model_path = model_path
        self.classes = np.asarray(classes)
        self.model = None

    def load(self, num_features: int) -> None:
        """加载 MultiRocket 模型。

        Args:
            num_features: 输入特征维度（MultiRocket 忽略此参数）。
        """
        self.model = joblib.load(self.model_path)

    def predict(self, batch: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """执行批量预测。

        Args:
            batch: 形状为 (batch_size, seq_len, n_features) 的输入数据。
            lengths: 形状为 (batch_size,) 的实际序列长度（MultiRocket 会自动处理）。

        Returns:
            (predictions, probabilities)
            - predictions: 形状为 (batch_size,) 的类别索引。
            - probabilities: 形状为 (batch_size, n_classes) 的类别概率。
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load()")

        # aeon 分类器期望输入为 (n_cases, n_channels, n_timepoints)
        if batch.ndim == 3:
            batch = batch.transpose(0, 2, 1)

        # MultiRocket 的 predict 返回类别标签
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
    def create_from_meta(meta_path: str, model_dir: str = "") -> MultiRocketPredictor:
        """从 meta.json 文件创建 MultiRocket 预测器。

        Args:
            meta_path: meta.json 文件路径。
            model_dir: 模型目录（如果 meta 中的路径是相对路径）。

        Returns:
            配置好的 MultiRocketPredictor 实例。
        """
        # 添加 radar_multirocket 到路径以导入 data_utils
        project_root = Path(__file__).resolve().parents[2]
        import sys
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
            # 如果 meta 中没有 model_files，使用默认路径
            meta_dir = Path(meta_path).parent
            model_path = meta_dir / "model.joblib"
        else:
            # 解析模型路径
            if model_dir:
                model_dir_path = Path(model_dir)
                candidate = Path(model_files[0])
                if candidate.is_file():
                    model_path = candidate
                else:
                    model_path = model_dir_path / model_files[0]
            else:
                meta_dir = Path(meta_path).parent
                model_path = meta_dir / model_files[0]

        if not model_path.is_file():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        return MultiRocketPredictor(
            model_path=str(model_path),
            classes=class_names,
        )

    @staticmethod
    def create(
        model_files: str,
        classes: list[str],
        model_dir: str = "",
    ) -> MultiRocketPredictor:
        """直接创建 MultiRocket 预测器。

        Args:
            model_files: 模型文件路径。
            classes: 类别名称列表。
            model_dir: 模型目录（如果 model_files 中的路径是相对路径）。

        Returns:
            配置好的 MultiRocketPredictor 实例。
        """
        if isinstance(model_files, str):
            model_files = [f.strip() for f in model_files.split(",") if f.strip()]

        model_file = model_files[0]

        if model_dir:
            model_dir_path = Path(model_dir)
            candidate = Path(model_file)
            if candidate.is_file():
                model_path = candidate
            else:
                model_path = model_dir_path / model_file
        else:
            model_path = Path(model_file)

        if not model_path.is_file():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        return MultiRocketPredictor(
            model_path=str(model_path),
            classes=classes,
        )
