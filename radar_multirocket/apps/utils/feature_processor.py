"""特征处理模块。

从解析的雷达目标数据中提取特征并进行归一化处理。
"""

import json
from pathlib import Path
from typing import Any

import numpy as np


# 特征映射：从解析器输出字段映射到特征字段
FEATURE_MAP = {
    "批号": "目标批号",
    "目标类型": "目标大类",
    "时间": "时间",
    "航迹历史": "航迹历史",
    "目标状态": "目标状态",
    "高（目标-滤波后）": "高（目标-滤波后）",
    "经（目标-滤波后）": "经（目标-滤波后）",
    "纬（目标-滤波后）": "纬（目标-滤波后）",
    "经（站址）": "经（站址）",
    "纬（站址）": "纬（站址）",
    "高（站址）": "高（站址）",
    "径向距离": "滤波径向距离",
    "方位": "滤波方位",
    "俯仰": "滤波俯仰",
    "点迹距离": "点迹距离",
    "点迹方位": "点迹方位",
    "点迹俯仰": "点迹俯仰",
    "全速度": "全速度",
    "径向速度": "径向速度",
    "方位速度": "方位速度",
    "俯仰速度": "俯仰速度",
    "航向": "航向",
    "目标信噪比": "目标信噪比",
    "多普勒展宽": "多普勒展宽",
    "JEM": "JEM",
    "RCS": "RCS",
    "目标流水号": "目标流水号",
    "识别信息大类": "目标大类",
    # 英文字段兼容
    "r_m": "r_m",
    "a_deg": "a_deg",
    "e_deg": "e_deg",
    "pr_m": "pr_m",
    "pa_deg": "pa_deg",
    "pe_deg": "pe_deg",
    "vel_m_s": "vel_m_s",
    "radial_vel_m_s": "radial_vel_m_s",
    "az_vel_deg_s": "az_vel_deg_s",
    "el_vel_deg_s": "el_vel_deg_s",
    "course_deg": "course_deg",
    "snr_db": "snr_db",
    "doppler": "doppler",
    "jem": "jem",
    "rcs_db": "rcs_db",
    "height_m": "height_m",
}

# 默认特征列配置
DEFAULT_FEATURE_COLS = [
    "径向距离",
    "点迹距离",
    "全速度",
    "径向速度",
    "方位速度",
    "俯仰速度",
    "加速度",
    "航向",
    "目标信噪比",
    "多普勒展宽",
    "JEM",
    "RCS",
    "点迹方位",
    "点迹俯仰",
]


class FeatureNormalizer:
    """特征归一化处理器。

    支持基于统计数据的均值-标准差归一化。
    """

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        """初始化归一化器。

        Args:
            mean: 均值数组，形状为 (n_features,)。
            std: 标准差数组，形状为 (n_features,)。
        """
        self.mean = mean
        self.std = std

    @classmethod
    def from_stats_file(
        cls, stats_path: str, feature_cols: list[str] | None = None
    ) -> "FeatureNormalizer":
        """从统计文件加载归一化参数。

        Args:
            stats_path: stats.json 文件路径。
            feature_cols: 特征列名称列表。

        Returns:
            配置好的归一化器实例。
        """
        feature_cols = feature_cols or DEFAULT_FEATURE_COLS
        path = Path(stats_path)
        if not path.is_file():
            return cls()

        with path.open("r", encoding="utf-8") as f:
            stats = json.load(f)

        mean = stats.get("mean")
        std = stats.get("std")

        if isinstance(mean, dict) and isinstance(std, dict):
            mean_arr = [mean.get(c, 0.0) for c in feature_cols]
            std_arr = [std.get(c, 1.0) for c in feature_cols]
        elif isinstance(mean, list) and isinstance(std, list):
            if len(mean) != len(feature_cols) or len(std) != len(feature_cols):
                raise ValueError("统计维度与特征维度不一致")
            mean_arr = mean
            std_arr = std
        else:
            raise ValueError("stats.json 结构不支持，仅支持 mean/std 的 list 或 dict")

        mean_arr = np.asarray(mean_arr, dtype=np.float32)
        std_arr = np.asarray(std_arr, dtype=np.float32)
        std_arr = np.where(std_arr == 0, 1.0, std_arr)
        return cls(mean_arr, std_arr)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """执行归一化。

        Args:
            x: 输入特征向量。

        Returns:
            归一化后的特征向量，如果未配置均值/标准差则返回原值。
        """
        if self.mean is None or self.std is None:
            return x
        return (x - self.mean) / self.std


class FeatureExtractor:
    """雷达目标特征提取器。

    从解析的 UDP 报文目标数据中提取特征。
    """

    def __init__(
        self,
        normalizer: FeatureNormalizer | None = None,
        feature_cols: list[str] | None = None,
        feature_map: dict[str, str] | None = None,
    ):
        """初始化特征提取器。

        Args:
            normalizer: 特征归一化器。
            feature_cols: 要提取的特征列名称列表。
            feature_map: 特征名称映射字典。
        """
        self.feature_cols = feature_cols or DEFAULT_FEATURE_COLS[:]
        self.feature_map = feature_map or FEATURE_MAP.copy()
        self.normalizer = normalizer or FeatureNormalizer()

    def extract(self, target: dict[str, Any]) -> np.ndarray | None:
        """从目标数据中提取特征。

        Args:
            target: 解析后的目标数据字典。

        Returns:
            提取并归一化后的特征向量，如果提取失败返回 None。
        """
        values = []
        for col in self.feature_cols:
            src_key = self.feature_map.get(col)
            if src_key is None:
                values.append(0.0)
            else:
                val = target.get(src_key, 0.0)
                # 特殊处理：某些字段需要单位转换
                if col in {"径向距离", "点迹距离", "全速度", "径向速度"}:
                    val = float(val) / 1000.0
                else:
                    val = float(val)
                values.append(val)

        vec = np.asarray(values, dtype=np.float32)
        return self.normalizer.normalize(vec)

    @property
    def num_features(self) -> int:
        """获取特征维度。"""
        return len(self.feature_cols)
